from numpy.linalg.linalg import solve
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch_geometric 
from gnn_dataset import GraphNodeDataset, instance_generator
from gnn_dataset import get_graph_from_obs
from gnn_policy import GNNPolicy, GNNNodeSelectionPolicy, GNNLtOPolicy
from acr_bb import ACRBBenv, DefaultBranchingPolicy, solve_bb, solve_bb_policy
import gzip
import pickle
from fcn_policy import FCNNodeSelectionLinearPolicy, FCNNodeDataset
from tqdm import tqdm
from observation import LinearObservation, Observation
import time

num_train = 10000
num_valid = 1000

train_data_path = '../../data/lto_train/'
valid_data_path = '../../data/lto_valid/'

def get_ogap(pred, target):
    f_pred = torch.norm(pred, 'fro', axis=1)
    f_target = torch.norm(target, 'fro', axis=1)
    ogap = ((f_pred-f_target)/f_target)*100
    ogap = ogap.mean()

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

TRAIN_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 128
POLICY_TYPE = 'gnn'
MODEL_PATH = '../trained_params/gnn_lto.model'
train_files = [str(path) for path in Path(train_data_path).glob('sample_*.pkl')]            
valid_files = [str(path) for path in Path(valid_data_path).glob('sample_*.pkl')]

print(len(train_files), len(valid_files))
sample_obs = pickle.load(gzip.open(train_files[0], 'rb'))[0]
M, N = sample_obs.variable_features.shape[0], sample_obs.antenna_features.shape[0]
print('M,N', M,N)



train_data = GraphNodeDataset(train_files)
train_loader = torch_geometric.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
valid_data = GraphNodeDataset(valid_files)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False) 

policy = GNNLtOPolicy()
policy = policy.train().to(DEVICE)
        
optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

loss_fn = nn.MSELoss() 
# training stage
total_data = 0
for _ in (range(TRAIN_EPOCHS)):
    mean_loss = 0
    n_samples_processed = 0
    
    for batch_data in tqdm(train_loader):
        batch, target = batch_data
        batch = batch.to(DEVICE)
        target = target.to(DEVICE)*1

        if POLICY_TYPE == 'gnn':
            batch_size = batch.num_graphs
        else:
            batch_size = batch.shape[0] 
        
        out = policy(batch.antenna_features, batch.edge_index, batch.edge_attr, batch.variable_features)
        print('shapes', out.shape, target.shape)
        loss = loss_fn(out.squeeze(), target.to(torch.float).squeeze())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss += loss.item() * batch_size
        ogap = get_ogap(out, target) 
    # validation
    for batch_data in tqdm(valid_loader):
        batch, target = batch_data
        batch = batch.to(DEVICE)
        target = target.to(DEVICE)*1

        if POLICY_TYPE == 'gnn':
            batch_size = batch.num_graphs
        else:
            batch_size = batch.shape[0] 
        
        out = policy(batch.antenna_features, batch.edge_index, batch.edge_attr, batch.variable_features)
        print('shapes', out.shape, target.shape)
        loss = loss_fn(out.squeeze(), target.to(torch.float).squeeze())

    
    # running tests
    print('Test Results, ogap={}'.format(ogap))
    torch.save(policy.state_dict(), MODEL_PATH)