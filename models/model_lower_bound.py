import torch
import torch.nn.functional as F
import torch_geometric
import gzip
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
import torch.nn as nn
from models.gnn_dataset import get_graph_from_obs

LEARNING_RATE = 0.001
NB_EPOCHS = 50
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMB_SIZE = 64

from models.setting import *

ANTENNA_NFEATS = 2
EDGE_NFEATS = 3
VAR_NFEATS = 1

NUM_TRAIN_H = 1000
N, M = 8,3
L = 5
FCN_FEATS_SIZE = 88

class FCNLowerBound(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
                    nn.Linear(FCN_FEATS_SIZE, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
            
#                     nn.Linear(512, 512),
#                     nn.BatchNorm1d(512),
#                     nn.ReLU(),
            
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU()
                    )
        self.z_module = torch.nn.Sequential(
                    nn.Linear(256, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    
                    nn.Linear(64, N),
                    nn.Sigmoid()
                    )
        self.power_module = torch.nn.Sequential(
                    nn.Linear(256, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    
                    nn.Linear(64, 1),
                    nn.ReLU()
                    )
        
    def forward(self, inp):
        latent = self.main(inp)
        return self.z_module(latent), self.power_module(latent)

class GNNLowerBound(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # ANTENNA EMBEDDING
        self.antenna_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(ANTENNA_NFEATS),
            torch.nn.Linear(ANTENNA_NFEATS, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(EDGE_NFEATS),
            torch.nn.Linear(EDGE_NFEATS, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
        )

        # USER EMBEDDING
        self.user_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(VAR_NFEATS),
            torch.nn.Linear(VAR_NFEATS, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
        )

        self.conv_user_to_antenna = BipartiteGraphConvolution()
        self.conv_antenna_to_user = BipartiteGraphConvolution()
        self.conv_antenna_to_user_final = BipartiteGraphConvolution()
        
        self.output_integral_module = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, 1, bias=False),
            torch.nn.Sigmoid(),
        )
        
        self.output_power_module = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, 1, bias=False),
        )

    def forward(self, obs):
        return self.pass_nn(obs.antenna_features, obs.edge_index, obs.edge_attr, obs.variable_features)
    
    def pass_nn(self, antenna_features, edge_indices, edge_features, user_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        antenna_features = self.antenna_embedding(antenna_features)
        edge_features = self.edge_embedding(edge_features)
        user_features = self.user_embedding(user_features)
        
        # Two half convolutions
        user_features = self.conv_antenna_to_user(antenna_features, edge_indices, edge_features, user_features)
        antenna_features = self.conv_user_to_antenna(user_features, reversed_edge_indices, edge_features, antenna_features)
        
        final_user_features = self.conv_antenna_to_user(antenna_features, edge_indices, edge_features, user_features)

        # A final MLP on the antenna features
        output_integral_relaxed = self.output_integral_module(antenna_features).squeeze(-1)
        
        output_power = self.output_power_module(final_user_features)
        output_power = output_power.sum()
        return (output_integral_relaxed, output_power)

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need 
    to provide the exact form of the messages being passed.
    """
    def __init__(self):
        super().__init__('add')
        
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE)
        )
        
        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(EMB_SIZE)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
        )

        
    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i) 
                                           + self.feature_module_edge(edge_features) 
                                           + self.feature_module_right(node_features_j))
        return output



class LowerBoundObservation(object):
    def __init__(self, H_complex, z_sol, z_mask):
        self.antenna_features  = None # np.zeros(N, 3) # three features for each antenna
        self.variable_features = None # np.zeros(M, 15)
        
        self.candidates        = [0] # np.arange(M)
        
        N, M = H_complex.shape
        
        # edge indices
        self.edge_index = np.stack((np.repeat(np.arange(N), M), np.tile(np.arange(M), N)))
        
        # edge features definition
        self.edge_features = np.zeros((M*N, EDGE_NFEATS))
        self.edge_features[:,0] = np.real(H_complex.reshape(-1))
        self.edge_features[:,1] = np.imag(H_complex.reshape(-1))
        self.edge_features[:,2] = np.abs(H_complex.reshape(-1))
        
        # antenna features definition
        self.antenna_features = np.zeros((N, ANTENNA_NFEATS))
        self.antenna_features[:,0] = z_sol
        self.antenna_features[:,1] = z_mask
        
        # user features definition
        self.variable_features = np.ones((M, VAR_NFEATS))
        
    def extract(self):
        return self
    

class LinearLowerBoundObservation(object):
    def __init__(self, H_complex, z_sol, z_mask):
        self.observation = np.concatenate((np.real(H_complex.reshape(-1)), 
                                            np.imag(H_complex.reshape(-1)),
                                            np.abs(H_complex.reshape(-1))))
        self.observation = np.concatenate((self.observation, 
                                         z_sol.reshape(-1),
                                         z_mask.reshape(-1)))
        
    def extract(self):
        return self
    

class LinearLowerBoundDataset(torch.utils.data.Dataset):
    def __init__(self, data_filepath):
        super().__init__()
        with open(data_filepath, 'rb') as handle:
            self.data = pickle.load(handle)

    def __len__(self):
        return len(self.data['out'])

    def __getitem__(self, index):
        sample_observation, np_target = self.data['in'][index], self.data['out'][index]
        
        target = []
        for i in range(len(np_target)):
            target.append(torch.tensor(np_target[i]))
        
        return sample_observation.observation, target
        
    
class GraphLowerBoundDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, data_filepath):
        super().__init__(root=None, transform=None, pre_transform=None)
        with open(data_filepath, 'rb') as handle:
            self.data = pickle.load(handle)

    def len(self):
        return len(self.data['out'])

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        
        sample_observation, np_target = self.data['in'][index], self.data['out'][index]
        
        target = []
        for i in range(len(np_target)):
            target.append(torch.tensor(np_target[i]))
            
        # not important
        candidates = torch.LongTensor(np.array([1,2,3], dtype=np.int32))
        candidate_choice = 1 
        sample_observation.variable_features = np.ones((M, VAR_NFEATS))
#         print(target[2])
        if target[2] < 0:
            target[2] = torch.FloatTensor(np.array(0.0))
        if target[2] > 1:
            target[2] = torch.FloatTensor(np.array(1.0))
        graph = BipartiteNodeData(sample_observation.antenna_features, sample_observation.edge_index, 
                                  sample_observation.edge_features, sample_observation.variable_features,
                                  candidates, candidate_choice)
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.antenna_features.shape[0] + sample_observation.variable_features.shape[0]
        
        return graph, target
    
def model_pass(model, H_complex=None, z_sol=None, z_mask=None):
    obs = LowerBoundObservation(H_complex, z_sol, z_mask)
    graph = get_graph_from_obs(obs, [1])
    a,b = model(graph)
    return a.detach().numpy(), b.detach().numpy()