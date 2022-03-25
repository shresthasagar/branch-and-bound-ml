import numpy as np
import itertools
from beamforming import solve_beamforming_with_selected_antennas
from multiprocessing import Pool
from as_omar import as_omar
# from as_bb import solve_bb, DefaultBranchingPolicy
# from as_bb import ASBBenv as Environment

from as_bb_test import solve_bb, DefaultBranchingPolicy
from as_bb_test import ASBBenv as Environment
import os
import pickle
import time
from pathlib import Path
from antenna_selection.observation import Observation

def solve_bb_pool(arguments):
    instance, max_ant = arguments
    return solve_bb(instance, max_ant=max_ant)


def solve_ml_pool(arguments):
    instance, w_optimal, optimal_objective, file_count, policy_filepath, max_ant, mask_heuristics, use_heuristics = arguments

    env = Environment(observation_function=Observation, epsilon=0.002)
    env.set_node_select_policy(node_select_policy_path=policy_filepath, policy_type='gnn')
    
    if use_heuristics:
        env.set_heuristic_solutions(mask_heuristics)

    env.reset(instance, max_ant=max_ant,  oracle_opt=w_optimal)

    branching_policy = DefaultBranchingPolicy()
    t1 = time.time()

    timestep = 0
    done = False
    time_taken = 0
    sum_label = 0
    node_counter = 0

    while timestep < 1000 and len(env.nodes)>0 and not done: 
        print('timestep {}'.format(timestep))
    
        env.fathom_nodes()
        if len(env.nodes) == 0:
            break
        node_id, node_feats, label = env.select_node()
        # print('observation instance', isinstance(node_feats, Observation))

        if len(env.nodes) == 0:
            break
        time_taken += time.time()-t1
        sum_label += label  
        node_counter += 1
        t1 = time.time()

        # print("Selection Time {}".format(time.time()-t1))
        # t1 = time.time()

        prune_node = env.prune(node_feats)
        # prune_node = False
        
        # print("Prune Decision Time {}".format(time.time()-t1))
        # t1 = time.time()

        if prune_node:
            env.delete_node(node_id)
            continue
        else:
            branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
            # try:
            #     done = env.push_children(branching_var, node_id)
            # except:
            #     print("exception occured")
            #     break
            done = env.push_children(branching_var, node_id)
        
        # print("Push children Time {}".format(time.time()-t1))
        
        timestep = timestep+1

    ml = np.linalg.norm(env.W_incumbent, 'fro')**2
    ogap = ((ml - optimal_objective)/optimal_objective)*100
    time_taken += time.time() - t1
    print('instance result', timestep, ogap, time_taken, sum_label, optimal_objective, ml)
    return timestep, ogap, time_taken, sum_label/node_counter
