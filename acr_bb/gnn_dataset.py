import torch
import torch.nn.functional as F
import torch_geometric
import gzip
import pickle
import numpy as np
from pathlib import Path


def instance_generator(M, N):
    while 1:
        yield np.random.randn(2,N,M)

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite` 
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, antenna_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice):
        super().__init__()

        if antenna_features is not None:
            self.antenna_features = torch.FloatTensor(antenna_features)
            self.edge_index = torch.LongTensor(edge_indices.astype(np.int64))
            self.edge_attr = torch.FloatTensor(edge_features)
            self.variable_features = torch.FloatTensor(variable_features)
            self.candidates = candidates
            self.nb_candidates = len(candidates)
            self.candidate_choices = candidate_choice

    def __inc__(self, key, value, *ags, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.antenna_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphNodeDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, target, _, _,_ = sample
        
        # We note on which variables we were allowed to branch, the scores as well as the choice 
        # taken by expert branching (relative to the candidates)
        candidates = torch.LongTensor(np.array([1,2,3], dtype=np.int32))
#         candidate_choice = sample_action_id
        candidate_choice = 1

        graph = BipartiteNodeData(sample_observation.antenna_features, sample_observation.edge_index, 
                                  sample_observation.edge_features, sample_observation.variable_features,
                                  candidates, candidate_choice)
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.antenna_features.shape[0] + sample_observation.variable_features.shape[0]
        
        return graph, target



class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action_id, sample_action_set = sample
        
        # We note on which variables we were allowed to branch, the scores as well as the choice 
        # taken by expert branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
#         candidate_choice = sample_action_id
        candidate_choice = torch.where(candidates == sample_action_id)[0][0]

        graph = BipartiteNodeData(sample_observation.antenna_features, sample_observation.edge_index, 
                                  sample_observation.edge_features, sample_observation.variable_features,
                                  candidates, candidate_choice)
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.antenna_features.shape[0] + sample_observation.variable_features.shape[0]
        
        return graph

class GraphDatasetFromObservation(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, obs):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.samples = obs

    def len(self):
        return len(self.samples)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        sample_observation = self.samples[index]
        
        # We note on which variables we were allowed to branch, the scores as well as the choice 
        # taken by expert branching (relative to the candidates)
        candidates = torch.LongTensor([1])
        candidate_choice = torch.LongTensor([1])
        graph = BipartiteNodeData(sample_observation.antenna_features, 
                                    sample_observation.edge_index, 
                                    sample_observation.edge_features,
                                    sample_observation.variable_features,
                                    candidates,
                                    candidate_choice)
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.antenna_features.shape[0] + sample_observation.variable_features.shape[0]
        
        return graph

class Experience(object):
    def __init__(self):
        self.current_state = []
        self.action = torch.Tensor([])
        self.reward = torch.Tensor([])
        self.next_state = []
        self.terminal = torch.BoolTensor([])
    
    def push(self, state, action, reward, next_state, done):
        self.current_state.append(state)
        self.action = torch.cat((self.action, torch.tensor([action])))
        self.reward = torch.cat((self.reward, torch.tensor([reward])))
        self.terminal = torch.cat((self.terminal, torch.BoolTensor([done])))
        self.next_state.append(next_state)
        
    def __len__(self):
        return len(self.current_state)

    def get_batch(self):
        current_state_set = GraphDatasetFromObservation(self.current_state)
        next_state_set = GraphDatasetFromObservation(self.next_state)
        current_state_loader = torch_geometric.data.DataLoader(current_state_set, batch_size=len(self.action))
        next_state_loader = torch_geometric.data.DataLoader(next_state_set, batch_size=len(self.action))
        return next(iter(current_state_loader)), self.action, self.reward, next(iter(next_state_loader)), self.terminal
        
