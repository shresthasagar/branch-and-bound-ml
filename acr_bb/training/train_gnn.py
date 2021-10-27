import torch
import torch.nn.functional as F
import torch_geometric
from gnn_dataset import BipartiteNodeData, GraphNodeDataset
from gnn_policy import GNNPolicy, BipartiteGraphConvolution
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import math 

LEARNING_RATE = 0.001
NB_EPOCHS = 150
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sig = nn.Sigmoid()

def process(policy, data_loader, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    targets_list = torch.Tensor([]).to(DEVICE)
    preds_list = torch.Tensor([]).to(DEVICE)
    with torch.set_grad_enabled(optimizer is not None):
        for batch_data in tqdm(data_loader):
            batch, target = batch_data
            batch = batch.to(DEVICE)
            target = target.to(DEVICE)*1
            
            logits = policy(batch.antenna_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            num_vars = int(batch.variable_features.shape[0]/batch.num_graphs)
            wts = torch.tensor([batch.variable_features[i*num_vars, 9] for i in range(batch.num_graphs)], dtype=torch.float32)
            wts = 2.68/wts
            wts = wts.to(DEVICE)
            logits = logits.reshape([batch.num_graphs, -1]).sum(dim=1)
#             wts = target*29
#             wts = wts+1
            bce = nn.BCEWithLogitsLoss(weight=wts)
#             bce = nn.BCEWithLogitsLoss()

            
            loss = bce(logits, target.to(torch.float))
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            predicted_bestindex = (sig(logits)>0.5)*1
            accuracy = sum(predicted_bestindex.reshape(-1) == target)
            
            targets_list = torch.cat((targets_list, target))
            preds_list = torch.cat((preds_list, predicted_bestindex))

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += float(accuracy)
            n_samples_processed += batch.num_graphs

    stacked = torch.stack((targets_list, preds_list), dim=1).to(torch.int)
    cmt = torch.zeros(2,2,dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    print(cmt)
    precision = cmt[1,1]/(cmt[0,1]+cmt[1,1])
    recall = cmt[1,1]/(cmt[1,0]+cmt[1,1])
    mean_acc = 2* (precision*recall)/(precision+recall)
    mean_loss /= n_samples_processed
#     mean_acc /= n_samples_processed
    return mean_loss, mean_acc


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output



positive_sample_files = [str(path) for path in Path('data/1_gnn_10k_positive_node_samples/').glob('sample_*.pkl')]
positive_trains = positive_sample_files[:int(0.8*len(positive_sample_files))]
positive_valids = positive_sample_files[int(0.8*len(positive_sample_files)):]

# negative_sample_files = ['data/1_gnn_10k_negative_node_samples/sample_'+str(i)+'.pkl' for i in sample_indices]
negative_sample_files = [str(path) for path in Path('data/1_gnn_10k_negative_node_samples/').glob('sample_*.pkl')]
negative_trains = negative_sample_files[:int(0.8*len(negative_sample_files))]
negative_valids = negative_sample_files[int(0.8*len(negative_sample_files)):]

random.shuffle(negative_trains)
negative_trains = negative_trains[:len(positive_trains)]

random.shuffle(negative_valids)
negative_valids = negative_valids[:len(positive_valids)]


train_files = positive_trains + negative_trains
valid_files = positive_valids + negative_valids

random.shuffle(train_files)


# random.shuffle(positive_trains)
# positive_sample_files = positive_sample_files[:100000]
# print('files loaded')

# # sample_indices = random.sample(range(0, 480000), 100000)
# negative_sample_files = ['data/1_gnn_10k_negative_node_samples/sample_'+str(i)+'.pkl' for i in sample_indices]
# negative_trains = negative_sample_files[:int(0.8*len(negative_sample_files))]
# negative_valid = negative_sample_files[int(0.8*len(negative_sample_files)):]

# imbalance_ratio = len(negative_sample_files)/len(positive_sample_files)

# random.shuffle(negative_sample_files)
# print(len(positive_sample_files), len(negative_sample_files))
# negative_sample_files = negative_sample_files[:len(positive_sample_files)]
# sample_files = positive_sample_files + negative_sample_files
# random.shuffle(sample_files)

# valid_sample_files = [str(path) for path in Path('node_samples/').glob('sample_*.pkl')]
# random.shuffle(valid_sample_files)
# valid_sample_files = valid_sample_files[:2000]

# train_files = sample_files[:int(0.8*len(sample_files))]
# valid_files = sample_files[int(0.8*len(sample_files)):]

train_data = GraphNodeDataset(train_files)
train_loader = torch_geometric.data.DataLoader(train_data, batch_size=128, shuffle=True)
valid_data = GraphNodeDataset(valid_files)
valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=128, shuffle=False)

policy = GNNPolicy().to(DEVICE)
optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

valids = []
trains = []
for epoch in range(NB_EPOCHS):
    print(f"Epoch {epoch+1}")
    
    train_loss, train_acc = process(policy, train_loader, optimizer)
    print(f"Train loss: {train_loss:0.3f}, f1-score {train_acc:0.3f}" )
    trains.append(train_acc)
    
    valid_loss, valid_acc = process(policy, valid_loader, None)
    print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )
    valids.append(valid_acc)

    torch.save(policy.state_dict(), 'data/trained_params_gnn4.pkl')