import numpy as np
from models.setting import DEBUG
from models.setting import ETA_EXP, NODE_DEPTH_INDEX, TASK, DATA_PATH, MODEL_PATH, RESULT_PATH, LOAD_MODEL, LOAD_MODEL_PATH, CLASS_IMBALANCE_WT, LAMBDA_ETA, DAGGER_NUM_ITER, DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER, DAGGER_NUM_VALID_EXAMPLES_PER_ITER, BB_MAX_STEPS


if TASK == 'antenna_selection':
    from antenna_selection.observation import Observation, LinearObservation
    from antenna_selection.as_bb import ASBBenv as Environment, DefaultBranchingPolicy, solve_bb

elif TASK == 'single_cast_beamforming':
    from single_beamforming.observation import Observation, LinearObservation
    from single_beamforming.acr_bb import ACRBBenv as Environment, DefaultBranchingPolicy, solve_bb

elif TASK == 'single_group_as_bm':
    from single_group_as_bm.observation import Observation, LinearObservation
    from single_group_as_bm.bb import BBenv as Environment, DefaultBranchingPolicy, solve_bb

elif TASK == 'robust_beamforming':
    from robust_beamforming.observation import Observation, LinearObservation
    from robust_beamforming.bb import BBenv as Environment, DefaultBranchingPolicy, solve_bb

import time
from models.helper import SolverException

def solve_bb_pool(arguments):
    try:
        instance, max_ant = arguments
        output = solve_bb(instance, max_ant)
    except SolverException as e:
        print('Solver Exception: ', e)
        return None, np.inf, 0, 0

    return output

def solve_ml_pool(arguments):
    instance, w_optimal, optimal_objective, file_count, policy_filepath, max_ant, mask_heuristics, use_heuristics = arguments

    env = Environment(observation_function=Observation, epsilon=0.002)
    env.set_node_select_policy(node_select_policy_path=policy_filepath, policy_type='gnn')
    
    if use_heuristics:
        env.set_heuristic_solutions(mask_heuristics)

    env.reset(instance, max_ant=max_ant,  oracle_opt=w_optimal)

    branching_policy = DefaultBranchingPolicy()
    t_total = time.time()

    timestep = 0
    done = False
    time_taken = 0
    sum_label = 0
    node_counter = 0

    misc_time = 0
    classification_time = 0
    branching_time = 0
    classified_nodes = 0
    branched_nodes = 0

    while timestep < 1000 and len(env.nodes)>0 and not done: 
        print('model tester timestep {},  U: {}, L: {}'.format(timestep, env.global_U, env.global_L))
        t1 = time.time()
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
        
        misc_time += time.time()-t1
        # print("Selection Time {}".format(time.time()-t1))
        t1 = time.time()

        prune_node = env.prune(node_feats)
        # prune_node = False

        classification_time += time.time()-t1
        classified_nodes += 1
        # print("Prune Decision Time {}".format(time.time()-t1))
        t1 = time.time()

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

        branching_time += time.time()-t1
        branched_nodes += 1
        # print("Push children Time {}".format(time.time()-t1))
        
        timestep = timestep+1
    
    ml = np.linalg.norm(env.W_incumbent, 'fro')**2
    # ml = env.global_U
    ogap = ((ml - optimal_objective)/optimal_objective)*100
    time_taken += time.time() - t_total
    print('instance result', timestep, ogap, time_taken, sum_label, optimal_objective, ml)
    return timestep, ogap, time_taken, sum_label/node_counter, (misc_time, classification_time, branching_time, classified_nodes, branched_nodes)

# def collect_data_instance(arguments):
#     instance, w_optimal, optimal_objective, file_count, policy_filepath, max_ant, mask_heuristics, use_heuristics = arguments

#     # instance, w_optimal, optimal_objective, file_count, policy_filepath = arguments
    
#     print('function {} started'.format(file_count))
#     #TODO: do the following with parameters not filename
#     # print('optimal ', w_optimal)
#     env = Environment(observation_function=Observation, epsilon=0.002)
#     env.set_node_select_policy(node_select_policy_path=policy_filepath, policy_type='gnn')
    
#     env.reset(instance, max_ant=max_ant,  oracle_opt=np.zeros(w_optimal.shape))

#     branching_policy = DefaultBranchingPolicy()
#     t1 = time.time()
#     timestep = 0
#     done = False
#     time_taken = 0
#     sum_label = 0
#     node_counter = 0
#     while timestep < 1000 and len(env.nodes)>0 and not done: 
#         print('collect data instance timestep {}'.format(timestep))
#         env.fathom_nodes()
#         if len(env.nodes) == 0:
#             break
#         node_id, node_feats, label = env.select_node()
#         if len(env.nodes) == 0:
#             break
#         time_taken += time.time()-t1
#         sum_label += label
#         node_counter += 1
#         t1 = time.time()

#         # print('Node id for pruning decision {}'.format(env.nodes[node_id].node_index))
#         prune_node = env.prune(node_feats)
#         # prune_node = False
#         if prune_node:
#             env.delete_node(node_id)
#             continue
#         else:
#             # print('Node id {}'.format(env.nodes[node_id].node_index))
#             last_id = len(env.nodes)

#             branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
#             try:
#                 done = env.push_children(branching_var, node_id)
#             except:
#                 break
            
#             # for i in range(last_id, len(env.nodes)):
#             #     print('Children {} z_mask {}, z_sol {}, l_angle {}, u angle {}'.format(env.nodes[i].node_index, env.nodes[i].z_mask, env.nodes[i].z_sol, env.nodes[i].l_angle, env.nodes[i].u_angle) )

#             # print('*********')
#             # print()
#         timestep = timestep+1
#         if env.is_terminal():
#             break

#     if node_counter < 1:
#         print('node counter null H {}, w_opt {}'.format(env.H_complex, w_optimal))
#     # ml = np.linalg.norm(env.W_incumbent, 'fro')**2
#     ml = env.global_U
#     # ml = np.linalg.norm(env.W_incumbent, 'fro')**2

#     ogap = ((ml - optimal_objective)/optimal_objective)*100
#     # if ogap>1:
#     #     print('H: {}, w_opt: {}, obj: {}, ml: {}'.format(env.H_complex, w_optimal, optimal_objective, ml))
#     #     debug_dict = {'H': env.H_complex,
#     #                   'w_opt': w_optimal,
#     #                   'obj': optimal_objective,
#     #                   'ml': ml}
#     #     with open('debug.pkl', 'wb') as f:
#     #         pickle.dump(debug_dict, f)

#     # time_taken += time.time() - t1
#     print('instance result', timestep, ogap, time_taken, sum_label, optimal_objective, ml)
#     # if ogap < -0.1:
#     #     print('obj: {}, ml: {}'.format(env.H_complex, w_optimal, optimal_objective, ml))
#     #     print('w_oracle: {}, w_ml: {}, z_oracl: {}, z_ml {}'.format(w_optimal[1], env.w_incumbent,  w_optimal[0], env.z_incumbent))
#     #     debug_dict = {'H': env.H_complex,
#     #                   'w_opt': w_optimal,
#     #                   'obj': optimal_objective,
#     #                   'ml': ml}
#     #     with open('debug.pkl', 'wb') as f:
#     #         pickle.dump(debug_dict, f)
#     # return order is timesteps, ogap, time (seconds), ratio of optimal nodes to total nodes 
#     return timestep, ogap, time_taken, sum_label/node_counter, None

