
import torch
import torch.nn as nn
import numpy as np
import time 
# from models.gnn_policy import GNNPolicy, GNNNodeSelectionPolicy
# from models.fcn_policy import FCNNodeSelectionLinearPolicy
# from models.gnn_dataset import get_graph_from_obs
from models.setting import MODEL_PATH
from single_group_as_bm.observation import *
# from single_group_as_bm.solve_relaxation import qp_relaxed, solve_relaxed_with_selected_antennas, cvxpy_relaxed
import numpy.ma as ma
from bb import *

min_bound_gap = 0.01

np.random.seed(seed = 100)


# N = [2, 4, 6, 8] 
# M = [4,8,12,16,32]
# max_ant = [2,4,6]
Ns = [2, 4, 6] 
Ms = [4,8,12,16,32]
Ls = [2,4,6]

combinations = []
# combinations.append((2,4,1))
# combinations.append((2,8,1))
# combinations.append((2,16,1))
# combinations.append((2,32,1))

# for N in Ns:
#     for M in Ms:
#         for max_ant in Ls:
#             if max_ant < N:       
#                 combinations.append((N,M,max_ant))
# combinations.append((8,12,6))
# combinations.append((6,12,4))
# combinations.append((6,16,4))
combinations.append((6,32,4))
combinations.append((8,12,6))
# combinations.append((2,32,1))

result_filename  = 'result.txt'
file_handle = open(result_filename, 'a')
file_handle.write('N,  M,  L,  u_avg,  time_avg,  timestep_avg \n')
file_handle.close()

result = []
for (N,M,max_ant) in combinations:
    u_avg = 0
    t_avg = 0
    tstep_avg = 0
    n_egs = 0
    for i in range(1):
        H = np.random.randn(N, M) + 1j*np.random.randn(N,M)    
        instance = np.stack((np.real(H), np.imag(H)), axis=0)
        try:
            z_inc, global_U, timesteps, t= solve_bb(instance, max_ant=max_ant, max_iter = 15000)
        except:
            continue
        u_avg += global_U
        t_avg += t
        n_egs += 1
        tstep_avg += timesteps
    try:
        u_avg /= n_egs
        t_avg /= n_egs 
        n_egs /= n_egs
        tstep_avg /= n_egs
    except:
        pass
    file_handle = open(result_filename, 'a')
    file_handle.write('{}, {}, {}, {}, {}, {}, {}\n'.format(N,M,max_ant, u_avg, t_avg, tstep_avg, n_egs))
    file_handle.close()

    result.append((N,M,max_ant, u_avg, t_avg, tstep_avg))
    print(N,M,max_ant, u_avg, t_avg, tstep_avg)