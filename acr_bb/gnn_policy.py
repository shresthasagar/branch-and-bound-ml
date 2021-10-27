import torch
import torch.nn.functional as F
import torch_geometric
import gzip
import pickle
import numpy as np
from pathlib import Path

LEARNING_RATE = 0.001
NB_EPOCHS = 50
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMB_SIZE = 64
ANTENNA_NFEATS = 9
EDGE_NFEATS = 3
VAR_NFEATS = 10

class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # CONSTRAINT EMBEDDING
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

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(VAR_NFEATS),
            torch.nn.Linear(VAR_NFEATS, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.antenna_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        

        # Two half convolutions
        # print('var', variable_features.shape, 'cons', constraint_features.shape, 'edge', reversed_edge_indices.shape, 'edge_f', edge_features.shape)
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output
    

class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # CONSTRAINT EMBEDDING
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

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(VAR_NFEATS),
            torch.nn.Linear(VAR_NFEATS, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, 1, bias=False),
        )

    def forward(self, state):
        constraint_features = state.antenna_features
        edge_indices = state.edge_index
        edge_features = state.edge_attr
        variable_features = state.variable_features

        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.antenna_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # sum_features = torch.sum(variable_features, dim=0)

        # A final MLP on the variable features
        # output = self.output_module(sum_features).squeeze(-1)
        # print(variable_features.shape)
        output = self.output_module(variable_features).squeeze(-1)

        return output
    

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
    
# from tqdm import tqdm

# def process(policy, data_loader, optimizer=None):
#     """
#     This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
#     """
#     mean_loss = 0
#     mean_acc = 0

#     n_samples_processed = 0
#     with torch.set_grad_enabled(optimizer is not None):
#         for batch in tqdm(data_loader):
#             batch = batch.to(DEVICE)
#             # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
#             logits = policy(batch.antenna_features, batch.edge_index, batch.edge_attr, batch.variable_features)
#             # Index the results by the candidates, and split and pad them
#             logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)

#             # Compute the usual cross-entropy classification loss
#             loss = F.cross_entropy(logits, torch.LongTensor(batch.candidate_choices))
#             if optimizer is not None:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
            
#             predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
#             accuracy = sum(predicted_bestindex.reshape(-1) == batch.candidate_choices)
# #             accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()

#             mean_loss += loss.item() * batch.num_graphs
#             mean_acc += float(accuracy)
#             n_samples_processed += batch.num_graphs

#     mean_loss /= n_samples_processed
#     mean_acc /= n_samples_processed
#     return mean_loss, mean_acc


# def pad_tensor(input_, pad_sizes, pad_value=-1e8):
#     """
#     This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
#     """
#     max_pad_size = pad_sizes.max()
#     output = input_.split(pad_sizes.cpu().numpy().tolist())
#     output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
#                           for slice_ in output], dim=0)
#     return output

# optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
# for epoch in range(NB_EPOCHS):
#     print(f"Epoch {epoch+1}")
    
#     train_loss, train_acc = process(policy, train_loader, optimizer)
#     print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}" )

#     valid_loss, valid_acc = process(policy, valid_loader, None)
#     print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )

# torch.save(policy.state_dict(), 'trained_params.pkl')

class GNNImitationPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # CONSTRAINT EMBEDDING
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

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(VAR_NFEATS),
            torch.nn.Linear(VAR_NFEATS, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.antenna_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        

        # Two half convolutions
        # print('var', variable_features.shape, 'cons', constraint_features.shape, 'edge', reversed_edge_indices.shape, 'edge_f', edge_features.shape)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)

        # A final MLP on the variable features
        output = self.output_module(constraint_features).squeeze(-1)
        return output
    