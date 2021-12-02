
from numpy.linalg.linalg import solve
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch_geometric 
from gnn_dataset import GraphNodeDataset, instance_generator
from gnn_dataset import get_graph_from_obs
from gnn_policy import GNNPolicy, GNNNodeSelectionPolicy
from acr_bb import ACRBBenv, DefaultBranchingPolicy, solve_bb, solve_bb_policy
import gzip
import pickle
from fcn_policy import FCNNodeSelectionLinearPolicy, FCNNodeDataset
from tqdm import tqdm
from observation import LinearObservation, Observation
import time


def test_bb(num_egs=10, policy=None, policy_type='gnn'):
    instances = instance_generator()
    ogap_avg = 0
    speedup = 0
    for i in range(num_egs):
        w_opt, f_opt, iters_opt, time_taken_opt = solve_bb_policy(next(instances), max_iter=1000)
        w, f, iters, time_taken = solve_bb_policy(next(instances), max_iter=1000, policy=policy, policy_type=policy_type)
        ogap_avg += (abs((f_opt-f)/(f_opt*num_egs))*100)
        speedup += time_taken_opt/(time_taken*num_egs)
        print('opt: {}, policy: {}, iters_opt: {}, iters: {}'.format(f_opt, f, iters_opt, iters))
    return ogap_avg, speedup


train_filepath = '../data/dagger_train'
valid_filepath = '../data/dagger_valid'

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

TRAIN_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 128
POLICY_TYPE = 'gnn'

train_files = [str(path) for path in Path(train_filepath).glob('sample_*.pkl')]            
valid_files = [str(path) for path in Path(valid_filepath).glob('sample_*.pkl')]

print(len(train_files), len(valid_files))
assert isinstance(pickle.load(gzip.open(train_files[0], 'rb'))[0], Observation)

train_data = GraphNodeDataset(train_files)
train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_data = GraphNodeDataset(valid_files)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False) 

policy = GNNNodeSelectionPolicy()
policy = policy.train().to(DEVICE)
        
optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# training stage
total_data = 0
for _ in tqdm(range(TRAIN_EPOCHS)):
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0
    targets_list = torch.Tensor([]).to(DEVICE)
    preds_list = torch.Tensor([]).to(DEVICE)
    t1 = time.time()
    for batch_data in tqdm(train_loader):
        # print('rest of the time', time.time() - t1)
        # t1 = time.time()
        batch, target = batch_data
        batch = batch.to(DEVICE)
        target = target.to(DEVICE)*1

        if POLICY_TYPE == 'gnn':
            batch_size = batch.num_graphs
            num_vars = int(batch.variable_features.shape[0]/batch_size)
            wts = torch.tensor([batch.variable_features[i*num_vars, 9] for i in range(batch_size)], dtype=torch.float32)
        else:
            batch_size = batch.shape[0] 
            wts = batch[:,-25]

        wts = 2.68/wts
        wts = wts.to(DEVICE)

        # print([batch.variable_features[i*num_vars, 9].item() for i in range(batch_size)], wts, target)
        wts = ((target)*7 + 1)*wts                   
        out = policy(batch, batch_size)
        bce = nn.BCELoss(weight=wts)        
        loss = bce(out.squeeze(), target.to(torch.float).squeeze())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        predicted_bestindex = (out>0.5)*1
        accuracy = sum(predicted_bestindex.reshape(-1) == target)

        targets_list = torch.cat((targets_list, target))
        preds_list = torch.cat((preds_list, predicted_bestindex))

        mean_loss += loss.item() * batch_size
        mean_acc += float(accuracy)
        n_samples_processed += batch_size
        # print(time.time()-t1)
        # t1 = time.time()
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

    # running tests
    ogap, speedup = test_bb(policy=policy)
    print('Test Results, ogap={}, speedup={}'.format(ogap, speedup))

