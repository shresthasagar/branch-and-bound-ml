from as_bb_test import solve_bb
from as_omar import *
from multiprocessing import Pool
import time
import pickle

def solve_bb_pool(arguments):
    instance, max_ant = arguments
    try:
        result = solve_bb(instance, max_ant=max_ant)
    except:
        result = None, None, None, None
    return result

N = [8,12,16]
M = [3,8,12]
L = [4,6,10,12]

combinations = []
num_egs = 30
for n in N:
    for m in M:
        if m <= n:
            for l in L:
                if l<n:
                    combinations.append((n,m,l))
                    
# Compute the ogap of Omar's method using the combinations

save_data_filepath = 'data/omar_eval/omar_evaluation_data.pkl'
save_result_filepath = 'data/omar_eval/omar_evaluation_result.pkl'

result = {'size':[], 'ogap':[], 'time':[], 'sol_rate':[]}
# data = {'instance': [], 'z_opt':[], 'power_opt':[], 'steps':[], 'time':[]}
data = {'size': [], 'data': []}
# Run optimal algorithm and save the result

np.random.seed(100)
for (n,m,l) in combinations:

# for (n,m,l) in combinations[5:]:
    ogap = 0
    time_avg = 0
    solved_instances = 0
    
    instances = np.random.randn(num_egs, 2, n, m)
    arguments_oracle = list(zip(list(instances), [l]*num_egs))

    with Pool(num_egs) as p:
        out_oracle = p.map(solve_bb_pool, arguments_oracle)
        print('pool ended')
    
    optimal_solution_list = [out_oracle[i][0] for i in range(len(out_oracle))]
    optimal_objective_list = [out_oracle[i][1] for i in range(len(out_oracle))]
    oracle_time = [out_oracle[i][3] for i in range(len(out_oracle))]
    
    solution_data = (instances, optimal_solution_list, optimal_objective_list, oracle_time)
    data['size'].append((n,m,l))
    data['data'].append(solution_data)
    
    omar_solved_instances = 0
    for i in range(num_egs):
        print("({}, {}, {}), eg: {}".format(n,m,l, i))
        H = instances[i,0,::] + 1j*instances[i,1,::]
        
        if optimal_objective_list[i] is not None:
            # run omar's method
            t1 = time.time()
            try:
                power, z = as_omar(H, max_ant=l)
            except:
                power = None
                z = None
            if power is not None:
                ogap += (power-optimal_objective_list[i])/optimal_objective_list[i] * 100
                omar_solved_instances += 1
                
            time_avg += time.time() - t1
            solved_instances += 1
    if omar_solved_instances >= 1:
        ogap = ogap/omar_solved_instances 
    else:
        ogap = 1e9
    if solved_instances >= 1:
        omar_solution_rate = omar_solved_instances/solved_instances
        time_avg = time_avg/solved_instances
    else:
        omar_solution_rate = 0
        time_avg = 0
    
    result['size'].append((n,m,l))
    result['ogap'].append(ogap)
    result['time'].append(time_avg)
    result['sol_rate'].append(omar_solution_rate)
    
    with open(save_result_filepath, 'wb') as handle:
        pickle.dump(result, handle)
    
    with open(save_data_filepath, 'wb') as handle:
        pickle.dump(data, handle)