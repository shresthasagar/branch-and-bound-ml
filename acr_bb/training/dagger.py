from observation import Observation, LinearObservation
import torch
import numpy as np
from acr_bb import ACRBBenv, DefaultBranchingPolicy
from gnn_policy import GNNPolicy
from tqdm import tqdm
import torch_geometric
import gzip
import pickle
from gnn_dataset import GraphNodeDataset, instance_generator


class TrainDagger(object):
    def __init__(self, train_instances, train_filepath='data/dagger_train/', valid_filepath='data/dagger_valid/'):
        # train instances should be a list of tuples (H, w_opt) 
        self.train_instances = train_instances 
        self.model = GNNPolicy()
        self.train_data = GraphNodeDataset(train_filepath)
        self.train_loader = torch_geometric.data.DataLoader(self.train_data, batch_size=128, shuffle=True)
        self.valid_data = GraphNodeDataset(valid_filepath)
        self.valid_loader = torch_geometric.data.DataLoader(self.valid_data, batch_size=128, shuffle=False)

        self.M = 8
        self.N = 4

        self.instances = instance_generator(self.M, self.N)
        self.data_collector = DataCollect( policy='oracle')

        data = self.data_collector.collect_data(self.instances, filepath=train_filepath, num_instances=10, )
        pass
    
    def train(self, train_epochs=10, iterations=100):
        for _ in tqdm(range(iterations)):
            # training stage
            for _ in range(train_epochs):
                pass

            # data collection stage

        


class DataCollect(object):
    def __init__(self, observation_function=Observation, policy='oracle',filepath=None):
        self.env = ACRBBenv(observation_function=observation_function, epsilon=0.0001, policy=policy)
        self.filepath = filepath
        self.counter = 0

    def collect_data(self, instances, num_instances=10, policy='oracle'):
        for _ in range(num_instances):
            instance = next(instances)
            w_opt = self.solve_bb(instance)
            data = self.collect_data_instance(instance, w_optimal=w_opt, prune_policy_path=policy)

    def collect_data_instance(self, instance, w_optimal=None, prune_policy_path='oracle'):
        #TODO: do the following with parameters not filename
        self.env.set_node_select_policy(policy=prune_policy_path)
        branching_policy = DefaultBranchingPolicy()
        # tuple of node observation, label
        data = []
        
        obs, actions, reward, done, _ = self.env.reset(instance, oracle_opt=w_optimal)        

        if done:
            return

        for i in range(1000):
            action_id = branching_policy.select_variable(obs, actions)
            obs, actions, reward, done, _ = self.env.step(action_id)
            if self.env.active_node.is_optimal:
                self.save_file((obs, 1))
            else:
                self.save_file((obs, 0))

            if done:
                

    def save_file(self, sample):
        filename = self.filepath + f'sample_{self.counter}.pkl'
        with gzip.open(filename, 'wb') as f:
            pickle.dump(sample, f)

