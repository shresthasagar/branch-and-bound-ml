import numpy as np
import torch
from gnn_dataset import BipartiteNodeData

def get_graph_from_obs(sample_observation, sample_action_set):
       
        sample_action_id = sample_action_set[0] # doen't matter won't be used
        # We note on which variables we were allowed to branch, the scores as well as the choice 
        # taken by expert branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action_id)[0][0]

        graph = BipartiteNodeData(sample_observation.antenna_features, sample_observation.edge_index, 
                                  sample_observation.edge_features, sample_observation.variable_features,
                                  candidates, candidate_choice)
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.antenna_features.shape[0] + sample_observation.variable_features.shape[0]
        
        return graph

