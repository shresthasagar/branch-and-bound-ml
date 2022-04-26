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

def enumeration_antenna_selection(H, max_ant):
    N, M = H.shape
    index_set  = np.arange(N)
    best_obj_val = 1000
    best_mask = np.zeros(N)
    total_nodes = 0
    for i in range(max_ant+1):
        print(i)
        for subset in itertools.combinations(index_set, i):
            mask = np.zeros(N)
            mask[list(subset)] = 1
            # mask[list((0,3,10))] = 1
            try:
                _, obj_val = solve_beamforming_with_selected_antennas(H, mask)
            except:
                obj_val = 1000
            print(subset, best_obj_val, obj_val)
            
            if best_obj_val > obj_val:
                best_obj_val = obj_val
                best_mask = mask.copy()
            total_nodes +=1
    return best_obj_val, best_mask, total_nodes

if __name__ == '__main__':
    N = 8
    M = 3
    max_ant = 5
    folder_path = '/scratch/sagar/Projects/combopt/branch-and-bound-ml/antenna_selection/validation_set/N={},M={},L={}_ftpl'.format(N,M,max_ant)
    if not os.path.isdir(folder_path):
        Path(folder_path).mkdir(exist_ok=True)

    result_filepath = os.path.join(folder_path, 'egs_set1.pkl') 

    ftpl = []
    nonftpl = []
    # MODEL_FILEPATH = '/scratch/sagar/Projects/combopt/branch-and-bound-ml/antenna_selection/trained_models/N={},M={},L={}/gnn1_iter_9'.format(N,M,max_ant)
    MODEL_FILEPATH = '/scratch/sagar/Projects/combopt/branch-and-bound-ml/antenna_selection/trained_models/N=8,M=3,L=5/gnn_ftpl_iter_30'.format(N,M,max_ant)

    num_egs = 30
    np.random.seed(seed = 100)
    num_nodes_eval = 0
    time_total = 0
    obj_bb = 0
    obj_omar = 0
    obj_opt = 0
    import time
    t1 = time.time()

    instances = np.random.randn(num_egs, 2, N, M)/np.sqrt(2)
    
    ######### ORACLE method #####################

    # t1 = time.time()
    # arguments_oracle = list(zip(list(instances), [max_ant]*num_egs))
    # with Pool(num_egs) as p:
    #     out_oracle = p.map(solve_bb_pool, arguments_oracle)
    #     print('pool ended')
    # # oracle_time = time.time() - t1

    # optimal_solution_list = [out_oracle[i][0] for i in range(len(out_oracle))]
    # optimal_objective_list = [out_oracle[i][1] for i in range(len(out_oracle))]
    # oracle_time = np.mean([out_oracle[i][3] for i in range(len(out_oracle))])

    # data = (instances, optimal_solution_list, optimal_objective_list, oracle_time)
    # with open(result_filepath, 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    ########### LOAD data ################################################

    with open(result_filepath, 'rb') as handle:
        data = pickle.load(handle)
        instances, optimal_solution_list, optimal_objective_list, oracle_time = data

    ########## OMAR'S Method #######################################
    mask_omar = []
    omar_time = 0
    for i in range(num_egs):
        H = instances[i,0,:,:] + 1j*instances[i,1,:,:]
        obj_bb += optimal_objective_list[i]
        t1 = time.time()
        obj, mask = as_omar(H, max_ant=max_ant)
        omar_time += time.time() - t1
        obj_omar += obj
        mask_omar.append(mask)
    omar_time = omar_time/num_egs
    print(omar_time)

    ########### GNN MODEL ###################################
    t1 = time.time()
    # H = np.stack((np.real(instances), np.imag(instances)), axis=1)
    print(instances.shape)
    arguments_ml = list(zip(list(instances), optimal_solution_list, optimal_objective_list, range(num_egs), [MODEL_FILEPATH]*num_egs, [max_ant]*num_egs, mask_omar))
    with Pool(num_egs) as p:
        out_ml = p.map(collect_data_instance, arguments_ml)
    # ml_time = time.time() - t1

    avg_ml_ogap = np.mean(np.array([out_ml[i][1] for i in range(len(out_ml))]))
    avg_ml_steps = np.mean(np.array([out_ml[i][0] for i in range(len(out_ml))]))
    ml_time = np.mean(np.array([out_ml[i][2] for i in range(len(out_ml))]))
    print('Model ogap: {}, time speedup: {}'.format(avg_ml_ogap, oracle_time/ml_time))
    print('Omar Ogap', (obj_omar-obj_bb)/obj_bb*100)
    print('BB time: {}, ML time: {} Omar time: {}'.format(oracle_time, ml_time, omar_time))

    t1 = time.time()
    # H = np.stack((np.real(instances), np.imag(instances)), axis=1)
    print(instances.shape)
    arguments_ml = list(zip(list(instances), optimal_solution_list, optimal_objective_list, range(num_egs), [MODEL_FILEPATH]*num_egs, [max_ant]*num_egs))
    with Pool(num_egs) as p:
        out_ml = p.map(collect_data_instance, arguments_ml)
    ml_time = time.time() - t1

    avg_ml_ogap = np.mean(np.array([out_ml[i][1] for i in range(len(out_ml))]))
    avg_ml_steps = np.mean(np.array([out_ml[i][2] for i in range(len(out_ml))]))

    print('Model ogap: {}, time speedup: {}'.format(avg_ml_ogap, oracle_time/ml_time))

    ####################################################################


            