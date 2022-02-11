from setting import NODE_DEPTH_INDEX, TASK, DATA_PATH

from antenna_selection.as_bb import ASBBenv as Environment, DefaultBranchingPolicy, solve_bb
if TASK == 'antenna_selection':
    from antenna_selection.observation import Observation, LinearObservation

elif TASK == 'single_cast_beamforming':
    from single_beamforming.observation import Observation, LinearObservation
    # from single_beamforming.acr_bb import ACRBBenv as Environment, DefaultBranchingPolicy, solve_bb

import torch
import torch.nn as nn
import numpy as np

from gnn_policy import GNNPolicy, GNNNodeSelectionPolicy
from tqdm import tqdm
import torch_geometric
import gzip
import pickle
from gnn_dataset import GraphNodeDataset, instance_generator
from pathlib import Path
from gnn_dataset import get_graph_from_obs
import shutil
import time
import os

from models.fcn_policy import FCNNodeSelectionLinearPolicy, FCNNodeDataset
from torch.utils.data import DataLoader

MAX_STEPS = 1000

class DataCollect(object):
    def __init__(self, observation_function=Observation, max_ant=None, policy='oracle',filepath=None, policy_type='gnn'):
        self.env = Environment(observation_function=observation_function, epsilon=0.002)
        self.max_ant = max_ant
        self.filepath = filepath
        self.counter = 0
        self.branching_policy = DefaultBranchingPolicy()
        self.policy_type = policy_type

    def collect_data(self, instances, num_instances=10, policy='oracle'):
        self.env.set_node_select_policy(node_select_policy_path=policy, policy_type=self.policy_type)
        sum_t, sum_ogap = 0, 0
        sum_t_default = 0
        sum_default_timesteps = 0
        sum_ml_time = 0
        for i in range(num_instances):
            print('collecting data instance {}'.format(i))
            instance = next(instances)
            # w_opt, t_default = solve_bb(instance)
            # print(w_opt.shape)
            optimal_solution,  optimal_objective, default_timesteps, t_default = solve_bb(instance, max_ant=self.max_ant, oracle_opt=np.random.randn(8))
            t, ogap, ml_time = self.collect_data_instance(instance, w_optimal=optimal_solution, optimal_objective=optimal_objective)
            sum_default_timesteps += default_timesteps
            sum_ml_time += ml_time
            sum_t += t
            sum_t_default += t_default
            sum_ogap += ogap
        return sum_t/sum_t_default, sum_ogap/num_instances, sum_default_timesteps/sum_ml_time

    # def collect_data_instance(self, instance, w_optimal=None):
    #     #TODO: do the following with parameters not filename
    #     self.env.reset(instance, max_ant=self.max_ant,  oracle_opt=w_optimal)
    #     t1 = time.time()
    #     timestep = 0
    #     done = False
    #     time_taken = 0
    #     while timestep < MAX_STEPS and len(self.env.nodes)>0 and not done:
    #         node_id, node_feats, label = self.env.select_node()

    #         if len(self.env.nodes) == 0:
    #             break
    #         time_taken += time.time()-t1
    #         self.save_file((node_feats, label))
    #         t1 = time.time()
    #         prune_node = self.env.prune(node_feats)
    #         if prune_node:
    #             self.env.delete_node(node_id)
    #             continue
    #         else:
    #             branching_var = self.branching_policy.select_variable(node_feats, self.env.action_set_indices)
    #             done = self.env.push_children(branching_var, node_id)
    #         timestep = timestep+1
    #     optimal = np.linalg.norm(w_optimal, 'fro')**2
    #     ml = np.linalg.norm(self.env.w_opt, 'fro')**2
    #     ogap = ((ml - optimal)/optimal)*100
    #     time_taken += time.time() - t1
    #     print(timestep, ogap, time_taken)
    #     return timestep, ogap, time_taken

    def collect_data_instance(self, instance, w_optimal=None, optimal_objective=None):
        #TODO: do the following with parameters not filename
        self.env.reset(instance, max_ant=self.max_ant,  oracle_opt=w_optimal)
        t1 = time.time()
        timestep = 0
        done = False
        time_taken = 0
        sum_label = 0
        while timestep < MAX_STEPS and len(self.env.nodes)>0 and not done:
            self.env.fathom_nodes()
            if len(self.env.nodes) == 0:
                break
            node_id, node_feats, label = self.env.select_node()

            if len(self.env.nodes) == 0:
                break
            time_taken += time.time()-t1
            sum_label += label
            self.save_file((node_feats, label))
            t1 = time.time()
            prune_node = self.env.prune(node_feats)
            if prune_node:
                self.env.delete_node(node_id)
                continue
            else:
                branching_var = self.branching_policy.select_variable(node_feats, self.env.action_set_indices)
                done = self.env.push_children(branching_var, node_id)
            timestep = timestep+1
        ml = np.linalg.norm(self.env.W_incumbent, 'fro')**2
        ogap = ((ml - optimal_objective)/optimal_objective)*100
        time_taken += time.time() - t1
        print('instance result', timestep, ogap, time_taken, sum_label, optimal_objective, ml)
        return timestep, ogap, time_taken

    def solve_bb(self, instance, w_optimal=None):
        self.env.reset(instance, oracle_opt=w_optimal)
        t1 = time.time()
        timestep = 0
        done = False
        while timestep < MAX_STEPS and len(self.env.nodes)>0 and not done:
            self.env.fathom_nodes()
            if len(self.env.nodes) == 0:
                break
            node_id, node_feats, label = self.env.select_node()

            if len(self.env.nodes) == 0:
                break
            # self.save_file((node_feats, label))
            
            # prune_node = self.env.prune(node_feats)
            prune_node=False
            if prune_node:
                self.env.delete_node(node_id)
                continue
            else:
                branching_var = self.branching_policy.select_variable(node_feats, self.env.action_set_indices)
                done = self.env.push_children(branching_var, node_id)
            timestep = timestep+1
        optimal = np.linalg.norm(w_optimal, 'fro')**2
        ml = np.linalg.norm(self.env.w_opt, 'fro')**2
        ogap = ((ml - optimal)/optimal)*100
        print(timestep, time.time()-t1)
        return timestep, self.env.w_opt.copy(),  time.time() - t1


    def save_file(self, sample):
        if self.filepath is not None:
            self.counter +=1
            filename = self.filepath + f'sample_{self.counter}.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump(sample, f)

