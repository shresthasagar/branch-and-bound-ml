
from setting import TASK

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
from torch.multiprocessing import Pool
import torch.multiprocessing as mp


from models.fcn_policy import FCNNodeSelectionLinearPolicy, FCNNodeDataset
from torch.utils.data import DataLoader

MAX_STEPS = 1000


class DataCollect(object):
    def __init__(self, observation_function=Observation, max_ant=None, policy='oracle',filepath=None, policy_type='gnn'):
        # env = Environment(observation_function=observation_function, epsilon=0.002)
        self.observation_function = observation_function
        self.max_ant = max_ant
        self.filepath = filepath
        self.branching_policy = DefaultBranchingPolicy()
        self.policy_type = policy_type
        self.node_select_policy = None


    def collect_data(self, instance_gen, num_instances=10, policy='oracle'):
        N, M = next(instance_gen).shape[1], next(instance_gen).shape[2]
        H = np.random.randn(num_instances, N, M) + 1j*np.random.randn(num_instances, N,M)    
        instances = np.stack((np.real(H), np.imag(H)), axis=1)
        

        arguments_oracle = list(zip(list(instances), [self.max_ant]*num_instances))
        print('starting first pool')
        with Pool(num_instances) as p:
            out_oracle = p.map(self.solve_bb_process, arguments_oracle)
            print('first pool ended')

        optimal_solution_list = [out_oracle[i][0] for i in range(len(out_oracle))]
        optimal_objective_list = [out_oracle[i][1] for i in range(len(out_oracle))]

        arguments_ml = list(zip(list(instances), optimal_solution_list, optimal_objective_list, range(num_instances), [policy]*num_instances))
        # arguments_ml = list(zip(list(instances), optimal_solution_list, optimal_objective_list, range(num_instances)))

        print('starting second pool')
        with Pool(num_instances) as p:
            out_ml = p.map(self.collect_data_instance, arguments_ml)
            # out_ml = p.map(self.dummy_collect_instance, arguments_ml)

            print('second pool ended')
        
        avg_oracle_steps = np.mean(np.array([out_oracle[i][2] for i in range(len(out_oracle))]))
        avg_oracle_time = np.mean(np.array([out_oracle[i][3] for i in range(len(out_oracle))]))
        avg_ml_time = np.mean(np.array([out_ml[i][0] for i in range(len(out_ml))]))
        avg_ml_ogap = np.mean(np.array([out_ml[i][1] for i in range(len(out_ml))]))
        avg_ml_steps = np.mean(np.array([out_ml[i][2] for i in range(len(out_ml))]))

        return avg_oracle_time/avg_ml_time, avg_ml_ogap, avg_oracle_steps/avg_ml_steps
        # return 0, 0, 0

    def collect_data_instance(self, arguments):
        instance, w_optimal, optimal_objective, file_count, policy_filepath = arguments
        print('function {} started'.format(file_count))
        #TODO: do the following with parameters not filename
        env = Environment(observation_function=self.observation_function, epsilon=0.002)
        env.set_node_select_policy(node_select_policy_path=policy_filepath, policy_type=self.policy_type)
        
        env.reset(instance, max_ant=self.max_ant,  oracle_opt=w_optimal)
        branching_policy = DefaultBranchingPolicy()
        t1 = time.time()
        timestep = 0
        done = False
        time_taken = 0
        sum_label = 0
        node_counter = 0
        while timestep < MAX_STEPS and len(env.nodes)>0 and not done: 
            # print('timestep {}'.format(timestep))
            env.fathom_nodes()
            if len(env.nodes) == 0:
                break
            node_id, node_feats, label = env.select_node()
            if len(env.nodes) == 0:
                break
            time_taken += time.time()-t1
            sum_label += label
            self.save_file((node_feats, label), file_count, node_counter)
            node_counter += 1
            t1 = time.time()

            prune_node = env.prune(node_feats)
            # prune_node = False
            if prune_node:
                env.delete_node(node_id)
                continue
            else:
                branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
                try:
                    done = env.push_children(branching_var, node_id)
                except:
                    break
            timestep = timestep+1

        ml = np.linalg.norm(env.W_incumbent, 'fro')**2
        ogap = ((ml - optimal_objective)/optimal_objective)*100
        time_taken += time.time() - t1
        print('instance result', timestep, ogap, time_taken, sum_label, optimal_objective, ml)
        return timestep, ogap, time_taken, sum_label/node_counter
    
    def solve_bb_process(self, tup):
        instance, max_ant = tup
        return solve_bb(instance, max_ant)
        

    def save_file(self, sample, file_count, node_counter):
        if self.filepath is not None:
            filename = self.filepath + f'sample_{file_count}_{node_counter}.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump(sample, f)

    def dummy_collect_instance(self, arguments):
        instance, w_optimal, optimal_objective, file_count = arguments
        print('started collect instance {}'.format(file_count))
        import time
        time.sleep(1)
        print('ended collect instance {}'.format(file_count))