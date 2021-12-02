import torch
import torch.nn.functional as F
import gzip
import pickle
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.setting import IN_FEATURES


class FCNDataset(Dataset):
    def __init__(self, sample_files):
        super().__init__()
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
        sample_observation, sample_action_id, _ = sample
        # out = sample_observation.observation - torch.mean(sample_observation.observation)
        return torch.tensor(sample_observation.observation, dtype=torch.float32), sample_action_id

class FCNNodeDataset(Dataset):
    def __init__(self, sample_files):
        super().__init__()
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
        sample_observation, label = sample
        # out = sample_observation.observation - torch.mean(sample_observation.observation)
        return torch.tensor(sample_observation.observation, dtype=torch.float32), torch.tensor(label)

class FCNNodeFakeDataset(Dataset):
    def __init__(self, sample_files):
        super().__init__()
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
        obs, label,_,_,_ = sample
        
        a = obs.observation[24:28] + 1j*obs.observation[28:32]
        angle = np.angle(a) 
        angle[angle<0] += 2*np.pi
        lb = obs.observation[24+3*8*4+2:126]
        ub = obs.observation[126:130]
        if np.random.rand(1)>0.8:
            mid = lb + np.random.rand(4)*(ub-lb)
            comp = np.exp(1j*mid)
        else:
            mid = np.random.rand(4)*2*np.pi
            comp = np.exp(1j*mid)
        if sum(mid<ub)+sum(mid>lb) == 8:
            label = 1
        else:
            label = 0
        # print(lb, ub)

        data = torch.cat((torch.tensor(np.real(mid)), torch.tensor(np.imag(mid)), torch.tensor(np.abs(mid)), torch.tensor(obs.observation)))
        # out = sample_observation.observation - torch.mean(sample_observation.observation)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label)


class FCNImitationDataset(Dataset):
    def __init__(self, sample_files):
        super().__init__()
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
        obs, _, _, target, _ = sample
        
        data = obs.observation[24:24+32*3]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(target.squeeze())


class FCNBranchingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(IN_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64,20),
            nn.BatchNorm1d(20),
            nn.ReLU(),

            nn.Linear(20, 4),
            nn.ReLU()


            # nn.Linear(1024, 512),
            # # nn.BatchNorm1d(512),
            # nn.ReLU(),
            
            # nn.Linear(512, 512),
            # # nn.BatchNorm1d(512),
            # nn.ReLU(),

            # # nn.Linear(128, 128),
            # # nn.BatchNorm1d(128),
            # # nn.ReLU(),

            # nn.Linear(512, 4),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # returns logits
        return self.main(x)


class FCNNodeSelectionPolicy(nn.Module):
    # Binary classifier as node selection policy
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(IN_FEATURES, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class FCNNodeSelectionLinearPolicy(nn.Module):
    # Binary classifier as node selection policy
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(IN_FEATURES, 1, bias=True),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),

            # # nn.Linear(512, 512),
            # # nn.BatchNorm1d(512),
            # # nn.ReLU(),
            
            # nn.Linear(512, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, num_graphs):
        return self.main(x)

class FCNImitationPolicy(nn.Module):
    # Binary classifier as node selection policy
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(32*3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 8),
        )

    def forward(self, x):
        return self.main(x)