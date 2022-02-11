from setting import NODE_DEPTH_INDEX, TASK, DATA_PATH, MODEL_PATH, CLASS_IMBALANCE_WT, ETA_EXP

from antenna_selection.as_bb import ASBBenv as Environment, DefaultBranchingPolicy, solve_bb
if TASK == 'antenna_selection':
    from antenna_selection.observation import Observation, LinearObservation

elif TASK == 'single_cast_beamforming':
    from single_beamforming.observation import Observation, LinearObservation
    # from single_beamforming.acr_bb import ACRBBenv as Environment, DefaultBranchingPolicy, solve_bb

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
from dagger_collect_data import DataCollect

MAX_STEPS = 10000
num_train_egs_per_iter = 1
num_valid_egs_per_iter = 1

class TrainDagger(object):
    def __init__(self, train_instances, train_filepath=os.path.join(DATA_PATH, 'dagger_train/'), valid_filepath=os.path.join(DATA_PATH, 'dagger_valid/'), policy_type='gnn'):
        """
        Runs dagger for imitating optimal node pruning policy 
        @params: 
            policy_type: one of {'linear', 'gnn'}
        """
        # train instances should be a list of tuples (H, w_opt) 
        self.train_instances = train_instances 
        if policy_type=='gnn':
            self.NodeDataset = GraphNodeDataset
            self.DataLoader = torch_geometric.data.DataLoader
            self.NodePolicy = GNNNodeSelectionPolicy
        
        if policy_type=='linear':
            self.NodeDataset = FCNNodeDataset
            self.DataLoader = DataLoader
            self.NodePolicy = FCNNodeSelectionLinearPolicy


            # self.policy = FCNNodeSelectionLinearPolicy()
            # self.train_data = FCNNodeDataset(train_filepath)
            # self.train_loader = DataLoader(self.train_data, batch_size=128, shuffle=True)
            # self.valid_data = FCNNodeDataset(valid_filepath)
            # self.valid_loader = DataLoader(self.valid_data, batch_size=128, shuffle=False)
        self.policy_type = policy_type
        self.policy = self.NodePolicy()
        self.train_data = self.NodeDataset(train_filepath)
        self.train_loader = self.DataLoader(self.train_data, batch_size=128, shuffle=True)
        self.valid_data = self.NodeDataset(valid_filepath)
        self.valid_loader = self.DataLoader(self.valid_data, batch_size=128, shuffle=False)
        
        self.DEVICE = 'cuda'
        self.policy = self.policy.to(self.DEVICE)

        self.M = 3
        self.N = 8
        self.max_ant = 5
        self.performance_list = []

        self.instances = instance_generator(self.M, self.N)

        if policy_type=='gnn':
            self.train_data_collector = DataCollect( observation_function=Observation, max_ant=self.max_ant, policy='oracle', filepath=train_filepath, policy_type=self.policy_type)
            self.valid_data_collector = DataCollect( observation_function=Observation, max_ant=self.max_ant, policy='oracle', filepath=valid_filepath, policy_type=self.policy_type)
        elif policy_type=='linear':
            self.train_data_collector = DataCollect(observation_function=LinearObservation, max_ant=self.max_ant, policy='oracle', filepath=train_filepath, policy_type=self.policy_type)
            self.valid_data_collector = DataCollect(observation_function=LinearObservation, max_ant=self.max_ant, policy='oracle', filepath=valid_filepath, policy_type=self.policy_type)
        else:
            raise NotImplementedError
            
        self.train_filepath = train_filepath
        self.valid_filepath = valid_filepath
        
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        pass
    
    def train(self, train_epochs=10, iterations=40):
        first_round = True
        best_t = 1000
        best_ogap = 1000
        for _ in tqdm(range(iterations)):
            torch.save(self.policy.eval().to('cpu').state_dict(), MODEL_PATH)
            if first_round:
                policy = 'oracle'
            else:
                print('selecting another ')
                policy = self.policy.eval().to('cpu')
            
            ## Uncomment this to delete the previously collected data at each iteration
            # path = Path(train_filepath)
            # shutil.rmtree(path)
            # path = Path(valid_filepath)
            # shutil.rmtree(path)
            # Path(train_filepath).mkdir(exist_ok=True)
            # Path(valid_filepath).mkdir(exist_ok=True)
            
            # data collection stage
            t, ogap, time_ratio = self.train_data_collector.collect_data(self.instances, num_instances=num_train_egs_per_iter, policy=policy)
            if not first_round:
                # if t<best_t :
                #     best_t = t
                if ogap < best_ogap:
                    best_ogap = ogap
                    best_t = t            

            print('ogap: {}, t: {}, best ogap: {}, best t: {}, time_ratio: {}'.format(ogap, t, best_ogap, best_t, time_ratio))
            if not first_round:
                self.performance_list.append((ogap, time_ratio))

            first_round = False
            self.valid_data_collector.collect_data(self.instances, num_instances=num_valid_egs_per_iter, policy=policy)
            
            train_files = [str(path) for path in Path(self.train_filepath).glob('sample_*.pkl')]            
            valid_files = [str(path) for path in Path(self.valid_filepath).glob('sample_*.pkl')]
            

            self.train_data = self.NodeDataset(train_files)
            self.train_loader = self.DataLoader(self.train_data, batch_size=128, shuffle=True)
            self.valid_data = self.NodeDataset(valid_files)
            self.valid_loader = self.DataLoader(self.valid_data, batch_size=128, shuffle=False) 

            self.policy = self.policy.train().to(self.DEVICE)

            # training stage
            total_data = 0
            for _ in tqdm(range(train_epochs)):
                mean_loss = 0
                mean_acc = 0
                n_samples_processed = 0
                targets_list = torch.Tensor([]).to(self.DEVICE)
                preds_list = torch.Tensor([]).to(self.DEVICE)
                for batch_data in (self.train_loader):
                    batch, target = batch_data
                    batch = batch.to(self.DEVICE)
                    target = target.to(self.DEVICE)*1

                    if self.policy_type == 'gnn':
                        batch_size = batch.num_graphs
                        num_vars = int(batch.variable_features.shape[0]/batch_size)
                        wts = torch.tensor([batch.variable_features[i*num_vars, NODE_DEPTH_INDEX] for i in range(batch_size)], dtype=torch.float32)
                    else:
                        batch_size = batch.shape[0] 
                        wts = batch[:,-25]

                    wts = 3/wts
                    wts = wts.to(self.DEVICE)

                    # print([batch.variable_features[i*num_vars, 9].item() for i in range(batch_size)], wts, target)
                    wts = ((target)*CLASS_IMBALANCE_WT + 1)*wts                   
                    out = self.policy(batch, batch_size)
                    bce = nn.BCELoss(weight=wts)        
                    loss = bce(out.squeeze(), target.to(torch.float).squeeze())
                    
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
                mean_acc = 2* (precision*recall)/(precision+recall)
                mean_loss /= n_samples_processed
                
                print("Train: precision:{}, recall:{}, f1-score:{}, loss: {}".format(precision, recall, mean_acc, mean_loss))


            # validation stage
            # mean_loss = 0
            # mean_acc = 0
            # n_samples_processed = 0
            # targets_list = torch.Tensor([]).to(self.DEVICE)
            # preds_list = torch.Tensor([]).to(self.DEVICE)
            # for batch_data in tqdm(self.valid_loader):
            #     batch, target = batch_data
            #     batch = batch.to(self.DEVICE)
            #     target = target.to(self.DEVICE)*1

            #     if self.policy_type == 'gnn':
            #         batch_size = batch.num_graphs
            #     else:
            #         batch_size = batch.shape[0] 

            #     out = self.policy(batch, batch_size)
            #     predicted_bestindex = (out>0.5)*1
            #     accuracy = sum(predicted_bestindex.reshape(-1) == target)
                
            #     targets_list = torch.cat((targets_list, target))
            #     preds_list = torch.cat((preds_list, predicted_bestindex))

            #     mean_acc += float(accuracy)
            #     n_samples_processed += batch_size

            # stacked = torch.stack((targets_list, preds_list.squeeze()), dim=1).to(torch.int)
            # cmt = torch.zeros(2,2,dtype=torch.int64)
            # for p in stacked:
            #     tl, pl = p.tolist()
            #     cmt[tl, pl] = cmt[tl, pl] + 1
            # print(cmt)
            # precision = cmt[1,1]/(cmt[0,1]+cmt[1,1])
            # recall = cmt[1,1]/(cmt[1,0]+cmt[1,1])
            # mean_acc = 2* (precision*recall)/(precision+recall)
            # print("Validataion: precision:{}, recall:{}, f1-score:{}".format(precision, recall, mean_acc))

            

            

if __name__=='__main__':
    instances = instance_generator(8, 3)
    
    train_filepath = os.path.join(DATA_PATH, 'dagger_train_gnn2/')
    valid_filepath = os.path.join(DATA_PATH,'dagger_valid_gnn2/')
    print('hello')

    if os.path.isdir(train_filepath):
        path = Path(train_filepath)
        shutil.rmtree(path)
    if os.path.isdir(valid_filepath):
        path = Path(valid_filepath)
        shutil.rmtree(path)
    
    Path(train_filepath).mkdir(exist_ok=True)
    Path(valid_filepath).mkdir(exist_ok=True)
    # data_collector = DataCollect(policy='oracle')
    # data_collector.collect_data(instances, num_instances=2)
    dagger = TrainDagger(instances, train_filepath=train_filepath, valid_filepath=valid_filepath, policy_type='gnn')
    dagger.train()
