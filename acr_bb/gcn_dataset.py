import torch
import torch.nn.functional as F
import torch_geometric
import gzip
import pickle
import numpy as np
from pathlib import Path



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