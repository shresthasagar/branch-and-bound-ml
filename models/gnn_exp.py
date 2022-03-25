from setting import ETA_EXP, NODE_DEPTH_INDEX, TASK, DATA_PATH, MODEL_PATH, RESULT_PATH, LOAD_MODEL, LOAD_MODEL_PATH, CLASS_IMBALANCE_WT, LAMBDA_ETA, DAGGER_NUM_ITER, DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER, DAGGER_NUM_VALID_EXAMPLES_PER_ITER, BB_MAX_STEPS

if TASK == 'antenna_selection':
    from antenna_selection.observation import Observation, LinearObservation
    from antenna_selection.as_bb_test import ASBBenv as Environment, DefaultBranchingPolicy, solve_bb

elif TASK == 'single_cast_beamforming':
    from single_beamforming.observation import Observation, LinearObservation
    from single_beamforming.acr_bb import ACRBBenv as Environment, DefaultBranchingPolicy, solve_bb

elif TASK == 'single_group_as_bm':
    from single_group_as_bm.observation import Observation, LinearObservation
    from single_group_as_bm.bb import BBenv as Environment, DefaultBranchingPolicy, solve_bb

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


np.random.seed(300)
N, M, max_ant = 6, 12, 2 
train_filepath = os.path.join(DATA_PATH, 'N={},M={},L={}/'.format(N,M,max_ant)) 
policy_type = 'gnn'

# train instances should be a list of tuples (H, w_opt) 
NodeDataset = GraphNodeDataset
DataLoader = torch_geometric.data.DataLoader
NodePolicy = GNNNodeSelectionPolicy

# training data is inside policy_data and the oracle solutions are inside oracle_data 
train_filepath = os.path.join(train_filepath, 'policy_data')
valid_filepath = os.path.join(train_filepath, 'valid_policy_data')


DEVICE = 'cuda'
if LOAD_MODEL:
    policy.load_state_dict(torch.load(LOAD_MODEL_PATH))
policy = policy.to(DEVICE)

performance_list = []

instances = instance_generator(M, N)

# Parameters for training
learning_rate = 0.001
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
train_epochs = 10
policy = NodePolicy()

model_folderpath = os.path.join(MODEL_PATH, 'gnn_exp')

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

model_filepath = os.path.join(model_folderpath, 'gnn_exp.model')
torch.save(policy.eval().to('cpu').state_dict(), model_filepath)

train_files = [str(path) for path in Path(train_filepath).glob('sample_*.pkl')]            
valid_files = [str(path) for path in Path(valid_filepath).glob('sample_*.pkl')]

train_data = NodeDataset(train_files)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

valid_data = NodeDataset(valid_files)
valid_loader = DataLoader(valid_data, batch_size=128, shuffle=True)

policy = policy.train().to(DEVICE)

# training stage
total_data = 0
for _ in tqdm(range(train_epochs)):
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0
    targets_list = torch.Tensor([]).to(DEVICE)
    preds_list = torch.Tensor([]).to(DEVICE)
    for batch_data in (train_loader):
        batch, target = batch_data
        batch = batch.to(DEVICE)
        target = target.to(DEVICE)*1

        if policy_type == 'gnn':
            batch_size = batch.num_graphs
            num_vars = int(batch.variable_features.shape[0]/batch_size)
            wts = torch.tensor([batch.variable_features[i*num_vars, NODE_DEPTH_INDEX] for i in range(batch_size)], dtype=torch.float32)
        else:
            batch_size = batch.shape[0] 
            wts = batch[:,-25]

        wts = 3/wts
        wts = wts.to(DEVICE)

        # print([batch.variable_features[i*num_vars, 9].item() for i in range(batch_size)], wts, target)
        wts = ((target)*CLASS_IMBALANCE_WT + 1)*wts                   
        out = policy(batch, batch_size)
        bce = nn.BCELoss(weight=wts)   

        try:
            F_w = bce(out.squeeze(), target.to(torch.float).squeeze())
        except:
            F_w = bce(out, target.to(torch.float))

        loss = F_w 

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
for batch_data in (valid_loader):
    batch, target = batch_data
    batch = batch.to(DEVICE)
    target = target.to(DEVICE)*1

    if policy_type == 'gnn':
        batch_size = batch.num_graphs
        num_vars = int(batch.variable_features.shape[0]/batch_size)
        wts = torch.tensor([batch.variable_features[i*num_vars, NODE_DEPTH_INDEX] for i in range(batch_size)], dtype=torch.float32)
    else:
        batch_size = batch.shape[0] 
        wts = batch[:,-25]

    wts = 3/wts
    wts = wts.to(DEVICE)

    wts = ((target)*CLASS_IMBALANCE_WT + 1)*wts                   
    out = policy(batch, batch_size)
    bce = nn.BCELoss(weight=wts)   

    try:
        F_w = bce(out.squeeze(), target.to(torch.float).squeeze())
    except:
        F_w = bce(out, target.to(torch.float))

    predicted_bestindex = (out>0.5)*1
    accuracy = sum(predicted_bestindex.reshape(-1) == target)

    targets_list = torch.cat((targets_list, target))
    preds_list = torch.cat((preds_list, predicted_bestindex))

    valid_mean_loss += loss.item() * batch_size
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

valid_loss = mean_loss
valid_acc = mean_acc

print("Valid: precision:{}, recall:{}, f1-score:{}, loss: {}, acc: {}".format(precision, recall, valid_mean_acc, valid_mean_loss, valid_acc))
