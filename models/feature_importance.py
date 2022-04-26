
from models.setting import ETA_EXP, NODE_DEPTH_INDEX, TASK, DATA_PATH, MODEL_PATH, RESULT_PATH, LOAD_MODEL, LOAD_MODEL_PATH, CLASS_IMBALANCE_WT, LAMBDA_ETA, DAGGER_NUM_ITER, DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER, DAGGER_NUM_VALID_EXAMPLES_PER_ITER, BB_MAX_STEPS

import torch
import torch.nn as nn
import numpy as np

from models.gnn_dataset import GraphNodeDatasetFromBipartiteNode
CLASS_IMBALANCE_WEIGHT = 1


class RandomizedDataset:
    def __init__(self, dataset):
        self.original_dataset = dataset
    
    def randomize_feature(self, feature_name='antenna', feature_indices=[]):
        """
        feature_name = one of 'antenna', 'user', or 'edge'
        feature_index = index of the feature 
        """
        
        assert len(feature_indices)>0, "Feature index should be provided"

        nodes = []
        rand_perm = np.random.permutation(len(self.original_dataset))
        
        try:
            for i in range(len(self.original_dataset)):
                new_node = (self.original_dataset[i][0].copy(), self.original_dataset[i][1])
                
                for feature_index in feature_indices:
                    if feature_name == 'antenna':
                        new_node[0].antenna_features[:,feature_index] = self.original_dataset[rand_perm[i]][0].antenna_features[:, feature_index].clone()
                    elif feature_name == 'user':
                        new_node[0].variable_features[:, feature_index] = self.original_dataset[rand_perm[i]][0].variable_features[:, feature_index].clone()                
                    elif feature_name == 'edge':
                        new_node[0].edge_attr[:, feature_index] = self.original_dataset[rand_perm[i]][0].edge_attr[:, feature_index].clone()

                nodes.append(new_node)
        except IndexError as e:
            print(e)
            print('Index of feaure {} is out of bounds for indices {}'.format(feature_name, feature_indices))
        return GraphNodeDatasetFromBipartiteNode(nodes)


def run_validation(valid_loader, policy, device='cuda'):
    valid_mean_loss = 0
    valid_mean_acc = 0
    n_samples_processed = 0
    targets_list = torch.Tensor([]).to(device)
    preds_list = torch.Tensor([]).to(device)
    for batch_data in (valid_loader):
        batch, target = batch_data
        batch = batch.to(device)
        target = target.to(device)*1

        batch_size = batch.num_graphs
        num_vars = int(batch.variable_features.shape[0]/batch_size)
        wts = torch.tensor([batch.variable_features[i*num_vars, NODE_DEPTH_INDEX] for i in range(batch_size)], dtype=torch.float32)
        
        wts = 3/wts
        wts = wts.to(device)

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

        valid_mean_loss += F_w.item() * batch_size
        valid_mean_acc += float(accuracy)
        n_samples_processed += batch_size

    stacked = torch.stack((targets_list, preds_list.squeeze()), dim=1).to(torch.int)
    cmt = torch.zeros(2,2,dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    precision = cmt[1,1]/(cmt[0,1]+cmt[1,1])
    recall = cmt[1,1]/(cmt[1,0]+cmt[1,1])

    valid_mean_acc = 2*(precision*recall)/(precision+recall)
    valid_mean_loss /= n_samples_processed

    # print(cmt)
    # print("Valid: precision:{}, recall:{}, f1-score:{}, loss: {}, acc: {}".format(precision, recall, valid_mean_acc, valid_mean_loss, valid_mean_loss))
    return precision, recall, valid_mean_loss, valid_mean_loss, cmt
    