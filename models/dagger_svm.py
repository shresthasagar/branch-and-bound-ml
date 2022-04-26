from setting import ETA_EXP, NODE_DEPTH_INDEX, TASK, DATA_PATH, MODEL_PATH, RESULT_PATH, LOAD_MODEL, LOAD_MODEL_PATH, CLASS_IMBALANCE_WT, LAMBDA_ETA, DAGGER_NUM_ITER, DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER, DAGGER_NUM_VALID_EXAMPLES_PER_ITER, BB_MAX_STEPS, DEVICE

if TASK == 'antenna_selection':
    from antenna_selection.observation import Observation, LinearObservation

elif TASK == 'single_cast_beamforming':
    from single_beamforming.observation import Observation, LinearObservation

elif TASK == 'single_group_as_bm':
    from single_group_as_bm.observation import Observation, LinearObservation

elif TASK == 'robust_beamforming':
    from robust_beamforming.observation import Observation, LinearObservation


import torch
import torch.nn as nn
import numpy as np

from gnn_policy import GNNPolicy, GNNNodeSelectionPolicy
from tqdm import tqdm
import torch_geometric
import gzip
import pickle
from gnn_dataset import GraphNodeDataset, instance_generator
from pathlib import Path
from gnn_dataset import get_graph_from_obs
import shutil
import time
import os

from models.fcn_policy import FCNNodeSelectionLinearPolicy, FCNNodeDataset
from torch.utils.data import DataLoader
from dagger_collect_data_multiprocess import DataCollect
import csv
from torch.distributions import Exponential

torch.set_num_threads(1)

def init_params_exp(policy, eta, device=DEVICE):
    sigma = []
    m = Exponential(torch.tensor([eta]))
    for param in policy.parameters():
        sigma.append(m.sample(param.shape).to(device))
    return sigma

class TrainDagger(object):
    def __init__(self, train_filepath=os.path.join(DATA_PATH, 'dagger_train/'), policy_type='gnn', result_filepath=RESULT_PATH, N=12, M=6, max_ant=None):
        """
        Runs dagger for imitating optimal node pruning policy 
        @params: 
            policy_type: one of {'linear', 'gnn'}
        """
        # train instances should be a list of tuples (H, w_opt) 
        if policy_type=='gnn':
            self.NodeDataset = GraphNodeDataset
            self.DataLoader = torch_geometric.data.DataLoader
            self.NodePolicy = GNNNodeSelectionPolicy
        
        if policy_type=='linear':
            self.NodeDataset = FCNNodeDataset
            self.DataLoader = DataLoader
            self.NodePolicy = FCNNodeSelectionLinearPolicy

        if not os.path.isdir(train_filepath):
            Path(train_filepath).mkdir(exist_ok=True)

        # training data is inside policy_data and the oracle solutions are inside oracle_data 
        self.train_filepath = os.path.join(train_filepath, 'policy_data')
        self.valid_filepath = os.path.join(train_filepath, 'valid_policy_data')


        if os.path.isdir(self.train_filepath):
            path = Path(self.train_filepath)
            shutil.rmtree(path)
        if os.path.isdir(self.valid_filepath):
            path = Path(self.valid_filepath)
            shutil.rmtree(path)
    
        Path(self.train_filepath).mkdir(exist_ok=True)
        Path(self.valid_filepath).mkdir(exist_ok=True)
        
        
        self.policy_type = policy_type
        self.policy = self.NodePolicy()

        self.train_data = self.NodeDataset(self.train_filepath)
        self.train_loader = self.DataLoader(self.train_data, batch_size=128, shuffle=True)

        self.valid_data = self.NodeDataset(self.valid_filepath)
        self.valid_loader = self.DataLoader(self.valid_data, batch_size=128, shuffle=True)

        if LOAD_MODEL:
            self.policy.load_state_dict(torch.load(LOAD_MODEL_PATH))
        self.policy = self.policy.to(DEVICE)

        self.M = M
        self.N = N
        self.max_ant = max_ant

        self.performance_list = []

        self.instances = instance_generator(self.M, self.N)

        self.result_filename  = os.path.join(result_filepath, 'result_M={}_N={}_L={}.txt'.format(self.M, self.N, self.max_ant))
        file_handle = open(self.result_filename, 'a')
        file_handle.write('iter_count, train_ogap, train_time_speedup, train_timestep_speedup, valid_ogap, valid_time_speedup, valid_timestep_speedup, train_acc, valid_acc, train_loss, valid_loss, train_fpr, valid_fpr, train_fnr, valid_fnr \n')
        file_handle.close()
        # self.csv_writer = csv.writer(file_handle)
        # self.csv_writer.writerow(('iter_count', 'ogap', 'speedup', 'timestep_speedup'))
        


        if policy_type=='gnn':
            self.train_data_collector = DataCollect(observation_function=Observation, 
                                                    max_ant=self.max_ant, 
                                                    policy='oracle', 
                                                    train_filepath=self.train_filepath, 
                                                    policy_type=self.policy_type,
                                                    oracle_solution_filepath=os.path.join(train_filepath, 'oracle_data'),
                                                    num_instances=DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER)

            self.valid_data_collector = DataCollect(observation_function=Observation, 
                                                    max_ant=self.max_ant, 
                                                    policy='oracle', 
                                                    train_filepath=self.valid_filepath, 
                                                    policy_type=self.policy_type,
                                                    oracle_solution_filepath=os.path.join(train_filepath, 'valid_oracle_data'),
                                                    num_instances=DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER)
        elif policy_type=='linear':
            self.train_data_collector = DataCollect(observation_function=LinearObservation, 
                                                    max_ant=self.max_ant, 
                                                    policy='oracle', 
                                                    train_filepath=self.train_filepath, 
                                                    policy_type=self.policy_type,
                                                    num_instances=DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER,
                                                    oracle_solution_filepath=os.path.join(train_filepath, 'oracle_data'))

            self.valid_data_collector = DataCollect(observation_function=LinearObservation, 
                                                    max_ant=self.max_ant, 
                                                    policy='oracle', 
                                                    train_filepath=self.valid_filepath, 
                                                    policy_type=self.policy_type,
                                                    oracle_solution_filepath=os.path.join(train_filepath, 'valid_oracle_data'),
                                                    num_instances=DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER)
        else:
            raise NotImplementedError
            
        
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        pass
    
    def train(self, train_epochs=10, iterations=30):
        if LOAD_MODEL:
            first_round = False
        else:
            first_round = True
        best_t = 1000
        best_ogap = 1000

        # FTRL
        # sigma = init_params_exp(self.policy, ETA_EXP, DEVICE)
        model_folderpath = os.path.join(MODEL_PATH, 'N={},M={},L={}'.format(N, M, max_ant))

        if not os.path.isdir(model_folderpath):
            Path(model_folderpath).mkdir(exist_ok = True)

        train_loss = 0
        valid_loss = 0
        train_acc = 0
        valid_acc = 0
        train_fpr = 0
        train_fnr = 0
        valid_fpr = 0
        valid_fnr = 0

        for iter_count in tqdm(range(DAGGER_NUM_ITER)):
            model_filepath = os.path.join(model_folderpath, 'gnn_iter_{}'.format(iter_count))
            torch.save(self.policy.eval().to('cpu').state_dict(), model_filepath)

            if first_round:
                policy = 'oracle'
            else:
                print('selecting another ')
                # policy = self.policy.eval().to('cpu')
                policy = model_filepath
            
            ## Uncomment this to delete the previously collected data at each iteration
            # path = Path(train_filepath)
            # shutil.rmtree(path)
            # path = Path(valid_filepath)
            # shutil.rmtree(path)
            # Path(train_filepath).mkdir(exist_ok=True)
            # Path(valid_filepath).mkdir(exist_ok=True)
            
            # data collection stage
            time_speedup, ogap, timestep_speedup = self.train_data_collector.collect_data(self.instances, 
                                                                        num_instances=DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER, 
                                                                        policy=policy)
            
            valid_time_speedup, valid_ogap, valid_timestep_speedup = self.valid_data_collector.collect_data(self.instances, 
                                                            num_instances=DAGGER_NUM_VALID_EXAMPLES_PER_ITER, 
                                                            policy=policy, 
                                                            train=False)
            if not first_round:
                # if time_speedup<best_t :
                #     best_t = time_speedup
                if ogap < best_ogap:
                    best_ogap = ogap
                    best_t = time_speedup            

            # TODO: add train loss, valid loss, valid ogap, speedup, timesteps_speedup
            print('ogap: {}, time_speedup: {}, best ogap: {}, best time_speedup: {}, timestep_speedup: {}'.format(ogap, time_speedup, best_ogap, best_t, timestep_speedup))
            print('VALID ogap: {}, time_speedup: {}, timestep_speedup: {}'.format(valid_ogap, valid_time_speedup, valid_timestep_speedup))

            # self.csv_writer.writerow((iter_count, ogap, time_speedup, timestep_speedup))
            file_handle = open(self.result_filename, 'a')
            file_handle.write('{:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f}, {:<.7f} \n'.format(iter_count, ogap, time_speedup, timestep_speedup, valid_ogap, valid_time_speedup, valid_timestep_speedup, train_acc, valid_acc, train_loss, valid_loss, train_fpr, valid_fpr, train_fnr, valid_fnr))
            file_handle.close()
            
            if not first_round:
                self.performance_list.append((ogap, timestep_speedup))

            # self.valid_data_collector.collect_data(self.instances, num_instances=DAGGER_NUM_VALID_EXAMPLES_PER_ITER, policy=policy)
            
            train_files = [str(path) for path in Path(self.train_filepath).glob('sample_*.pkl')]            
            valid_files = [str(path) for path in Path(self.valid_filepath).glob('sample_*.pkl')]
            

            self.train_data = self.NodeDataset(train_files)
            self.train_loader = self.DataLoader(self.train_data, batch_size=128, shuffle=True)

            self.valid_data = self.NodeDataset(valid_files)
            self.valid_loader = self.DataLoader(self.valid_data, batch_size=128, shuffle=True)


            self.policy = self.policy.train().to(DEVICE)

            first_round = False
            # training stage
            total_data = 0
            for _ in tqdm(range(train_epochs)):
                mean_loss = 0
                mean_acc = 0
                n_samples_processed = 0
                targets_list = torch.Tensor([]).to(DEVICE)
                preds_list = torch.Tensor([]).to(DEVICE)
                for batch_data in (self.train_loader):
                    batch, target = batch_data
                    batch = batch.to(DEVICE)
                    target = target.to(DEVICE)*1

                    if self.policy_type == 'gnn':
                        batch_size = batch.num_graphs
                        num_vars = int(batch.variable_features.shape[0]/batch_size)
                        wts = torch.tensor([batch.variable_features[i*num_vars, NODE_DEPTH_INDEX] for i in range(batch_size)], dtype=torch.float32)
                    else:
                        batch_size = batch.shape[0] 
                        wts = batch[:,-25]

                    wts = 1/wts
                    wts = wts.to(DEVICE)

                    # print([batch.variable_features[i*num_vars, 9].item() for i in range(batch_size)], wts, target)
                    wts = ((target)*CLASS_IMBALANCE_WT + 1)*wts                   
                    out = self.policy(batch, batch_size)
                    bce = nn.BCELoss(weight=wts)   

                    # Regularization parameter to ensure convergence in non-convex online learning
                    R_w = 0
                    # # Uncomment the following two lines to include FTPL regularization 
                    # for (param, sig) in zip(self.policy.parameters(), sigma):
                    #     R_w += torch.dot(param.flatten(), sig.flatten())
                    # print(out.shape, target.shape)
                    # print(out, target)
                    # print(out.squeeze().shape, target.to(torch.float).squeeze().shape)
                    # print('len batch', batch_size)
                    for ind in range(len(out)):
                        if np.isnan(out.cpu().detach().numpy()[ind]):
                            num_vars = int(batch.variable_features.shape[0]/batch_size)
                            num_ants = int(batch.antenna_features.shape[0]/batch_size)
                            print('variable features', batch.variable_features[ind*num_vars])
                            print('antenna features', batch.antenna_features[ind*num_ants])
                            # wts = torch.tensor([batch.variable_features[i*num_vars, NODE_DEPTH_INDEX] for i in range(batch_size)], dtype=torch.float32)
                    try:
                        F_w = bce(out.squeeze(), target.to(torch.float).squeeze())
                    except:
                        F_w = bce(out, target.to(torch.float))

                    # print("Fw and Rw", F_w.item(), R_w.item())x`
                    loss = F_w + LAMBDA_ETA*R_w
                    
                    if self.optimizer is not None:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    predicted_bestindex = (out>0.5)*1
                    accuracy = sum(predicted_bestindex.reshape(-1) == target)
                    
                    targets_list = torch.cat((targets_list, target))
                    preds_list = torch.cat((preds_list, predicted_bestindex))

                    mean_loss += loss.item() * batch_size
                    mean_acc += float(accuracy)
                    n_samples_processed += batch_size
                total_data = n_samples_processed
                stacked = torch.stack((targets_list, preds_list.squeeze()), dim=1).to(torch.int)
                cmt = torch.zeros(2,2,dtype=torch.int64)
                for p in stacked:
                    tl, pl = p.tolist()
                    cmt[tl, pl] = cmt[tl, pl] + 1
                print(cmt)
                precision = cmt[1,1]/(cmt[0,1]+cmt[1,1])
                recall = cmt[1,1]/(cmt[1,0]+cmt[1,1])

                train_fpr = cmt[0,1]/(cmt[0,0] + cmt[0,1])
                train_fnr = cmt[1,0]/(cmt[1,0] + cmt[1,1])

                mean_acc = 2* (precision*recall)/(precision+recall)
                mean_loss /= n_samples_processed
                print("Train: precision:{}, recall:{}, f1-score:{}, loss: {}, acc: {}".format(precision, recall, mean_acc, mean_loss, mean_acc))
            train_loss = mean_loss
            train_acc = mean_acc

            valid_mean_loss = 0
            valid_mean_acc = 0
            n_samples_processed = 0
            targets_list = torch.Tensor([]).to(DEVICE)
            preds_list = torch.Tensor([]).to(DEVICE)
            for batch_data in (self.valid_loader):
                batch, target = batch_data
                batch = batch.to(DEVICE)
                target = target.to(DEVICE)*1

                if self.policy_type == 'gnn':
                    batch_size = batch.num_graphs
                    num_vars = int(batch.variable_features.shape[0]/batch_size)
                    wts = torch.tensor([batch.variable_features[i*num_vars, NODE_DEPTH_INDEX] for i in range(batch_size)], dtype=torch.float32)
                else:
                    batch_size = batch.shape[0] 
                    wts = batch[:,-25]

                wts = 3/wts
                wts = wts.to(DEVICE)

                wts = ((target)*CLASS_IMBALANCE_WT + 1)*wts                   
                out = self.policy(batch, batch_size)
                bce = nn.BCELoss(weight=wts)   
                
                try:
                    F_w = bce(out.squeeze(), target.to(torch.float).squeeze())
                except:
                    F_w = bce(out, target.to(torch.float))

                predicted_bestindex = (out>0.5)*1
                accuracy = sum(predicted_bestindex.reshape(-1) == target)
                
                targets_list = torch.cat((targets_list, target))
                preds_list = torch.cat((preds_list, predicted_bestindex))

                valid_mean_loss += F_w.item() * batch_size
                valid_mean_acc += float(accuracy)
                n_samples_processed += batch_size
            total_data = n_samples_processed
            stacked = torch.stack((targets_list, preds_list.squeeze()), dim=1).to(torch.int)
            cmt = torch.zeros(2,2,dtype=torch.int64)
            for p in stacked:
                tl, pl = p.tolist()
                cmt[tl, pl] = cmt[tl, pl] + 1
            print(cmt)
            precision = cmt[1,1]/(cmt[0,1]+cmt[1,1])
            recall = cmt[1,1]/(cmt[1,0]+cmt[1,1])

            valid_fpr = cmt[0,1]/(cmt[0,0] + cmt[0,1])
            valid_fnr = cmt[1,0]/(cmt[1,0] + cmt[1,1])

            valid_mean_acc = 2* (precision*recall)/(precision+recall)
            valid_mean_loss /= n_samples_processed

            valid_loss = valid_mean_loss
            valid_acc = valid_mean_acc
            
            print("Valid: precision:{}, recall:{}, f1-score:{}, loss: {}, acc: {}".format(precision, recall, valid_mean_acc, valid_mean_loss, valid_acc))


            

if __name__=='__main__':
    np.random.seed(100)
    N, M, max_ant = 8,6,4
    train_filepath = os.path.join(DATA_PATH, 'corrected_N={},M={},L={}/'.format(N,M,max_ant))
    
    # train_filepath = os.path.join(DATA_PATH, 'dagger_train_gnn_ftpl_N={},M={},L={}/'.format(N,M,max_ant))
   

    # if os.path.isdir(train_filepath):
    #     path = Path(train_filepath)
    #     shutil.rmtree(path)
    
    # Path(train_filepath).mkdir(exist_ok=True)
    # Path(os.path.join(train_filepath, '/positives'))
    # Path(os.path.join(train_filepath, '/negatives'))

    # data_collector = DataCollect(policy='oracle')
    # data_collector.collect_data(instances, num_instances=2)
    dagger = TrainDagger(train_filepath=train_filepath, policy_type='gnn', N=N, M=M, max_ant=max_ant)
    dagger.train()
