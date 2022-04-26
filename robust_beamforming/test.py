import numpy as np
import itertools
from multiprocessing import Pool

import os
import pickle
import time
from pathlib import Path
from models.model_tester import *
from models.setting import MODEL_PATH, DATA_PATH, VALIDATION_PATH, TASK
from robust_beamforming.convex_baseline import ReweightedPenalty

N,M,max_ant = 10,4,3
np.random.seed(seed = 1000)
num_egs = 30

ftpl2 = []
nonftpl = []

folder_path = os.path.join(VALIDATION_PATH, 'N={},M={},L={}'.format(N,M,max_ant))
# folder_path = '/scratch/sagar/Projects/combopt/branch-and-bound-ml/single_group_as_bm/validation_set/N={},M={},L={}'.format(N,M,max_ant)

if not os.path.isdir(folder_path):
    Path(folder_path).mkdir(exist_ok=True)

result_filepath = os.path.join(folder_path, 'egs_set_valid_30.pkl') 

MODEL_FOLDERPATH = os.path.join(MODEL_PATH, 'N={},M={},L={}'.format(N,M,max_ant))

num_nodes_eval = 0
time_total = 0
obj_bb = 0
obj_omar = 0
obj_opt = 0
t1 = time.time()

instances = np.random.randn(num_egs, 2, N, M)

######## ORACLE method #####################

t1 = time.time()
arguments_oracle = list(zip(list(instances), [max_ant]*num_egs))
with Pool(10) as p:
    out_oracle = p.map(solve_bb_pool, arguments_oracle)
    print('pool ended')

# out_oracle =[]
# for i in range(num_egs):
#     print('eg {}'.format(i))
#     try:
#         output = solve_bb_pool(arguments_oracle[i])
#         out_oracle.append(output)        
#     except SolverException as e:
#         print('Solver Exception: ', e)
#         out_oracle.append((None, np.inf, 0, 0))
    
# Prune away the problem instances that were not feasible (could not be solved)
for i in range(len(out_oracle)-1, -1, -1):
    if out_oracle[i][1] == np.inf:
        del out_oracle[i]
        instances = np.concatenate((instances[:i,::], instances[i+1:,::]), axis=0)

optimal_solution_list = [out_oracle[i][0] for i in range(len(out_oracle))]
optimal_objective_list = [out_oracle[i][1] for i in range(len(out_oracle))]
oracle_timesteps = np.mean([out_oracle[i][2] for i in range(len(out_oracle))])
oracle_time = np.mean([out_oracle[i][3] for i in range(len(out_oracle))])


data = (instances, optimal_solution_list, optimal_objective_list, oracle_timesteps, oracle_time)
with open(result_filepath, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    

########### LOAD data ################################################

with open(result_filepath, 'rb') as handle:
    data = pickle.load(handle)
    instances, optimal_solution_list, optimal_objective_list, oracle_timesteps, oracle_time = data

########## OMAR'S Method #######################################
mask_omar = [0]*N
obj_omar = []
omar_time = 0
for i in range(len(instances)):
    H = instances[i,0,:,:] + 1j*instances[i,1,:,:]
    baseline = ReweightedPenalty(H=H, max_ant=max_ant)
    obj_bb += optimal_objective_list[i]
    t1 = time.time()
    obj, mask = baseline.solve()
    omar_time += time.time() - t1
    obj_omar.append(obj)
    mask_omar.append(mask)
omar_time = omar_time/len(instances)
print(omar_time)

mean_obj_omar = 0
mean_obj_bb = 0
num_success = 0
for i in range(len(obj_omar)):
    if obj_omar[i] is not None:
        mean_obj_omar += obj_omar[i]
        mean_obj_bb += optimal_objective_list[i]
        num_success += 1
mean_obj_omar /= num_success
mean_obj_bb /= num_success
success_rate = num_success/len(obj_omar)
omar_ogap = (mean_obj_omar - mean_obj_bb)/mean_obj_bb*100
########## GNN MODEL ###################################

model_ids = [7]

### FTPL ###
mask_omar = np.zeros(N)
for i in model_ids:
    MODEL_FILEPATH = os.path.join(MODEL_FOLDERPATH, 'gnn_1_iter_{}'.format(i))
    t1 = time.time()
    arguments_ml = list(zip(list(instances), 
                            optimal_solution_list, 
                            optimal_objective_list, 
                            range(len(instances)), 
                            [MODEL_FILEPATH]*len(instances), 
                            [max_ant]*len(instances), 
                            [mask_omar]*len(instances), 
                            [False]*len(instances)))
    
    out_ml = []
    for j in range(len(instances)):
        print('eg {}'.format(i))
        try:
            output = solve_ml_pool(arguments_ml[j])
            out_ml.append(output)
        except:
            pass
    
    avg_ml_ogap = np.mean(np.array([out_ml[i][1] for i in range(len(out_ml))]))
    avg_ml_steps = np.mean(np.array([out_ml[i][0] for i in range(len(out_ml))]))
    ml_time = np.mean(np.array([out_ml[i][2] for i in range(len(out_ml))]))
    ftpl2.append((avg_ml_ogap, avg_ml_steps, ml_time, oracle_timesteps, omar_ogap))
    
result_data = (out_ml, mask_omar, obj_omar, ftpl2)
ml_result_path = os.path.join(folder_path, 'egs_set_valid_30_result.pkl') 
with open(ml_result_path, 'wb') as handle:
    pickle.dump(result_data, handle, protocol=pickle.HIGHEST_PROTOCOL)    
