import gzip
import pickle
import numpy as np
from pathlib import Path
import torch
from  acr_bb import Observation, ACRBBenv, DefaultBranchingPolicy, RandomPolicy, LinearObservation
from fcn_training import FCNDataset, FCNBranchingPolicy, FCNNodeSelectionPolicy 

Positive_sample_path = 'data/2_fcn_10k_positive_node_samples/'
Negative_sample_path = 'data/2_fcn_10k_negative_node_samples/'

def collect_node_samples(observation_function=Observation, N=8, M=4, max_samples=10000, max_episodes=10000, collect_positive=False):
    expert_prob = 0.5

    def instance_generator(M, N):
        while 1:
            yield np.random.randn(2,N,M)

    # instances = np.random.randn(max_samples, 2, N, M)
    instances = instance_generator(M,N)

    env = ACRBBenv(observation_function=observation_function, node_select_policy_path='default')

    expert_policy = DefaultBranchingPolicy()
    random_policy = RandomPolicy() 

    episode_counter, sample_counter, negative_counter = 0, 0, 0

    Path(Positive_sample_path).mkdir(exist_ok=True)
    Path(Negative_sample_path).mkdir(exist_ok=True)

    # We will solve problems (run episodes) until we have saved enough samples
    max_samples_reached = False

    while not max_samples_reached:
        episode_counter += 1
        
        observation_list = []
        node_indices = []
        env = ACRBBenv(observation_function=observation_function, node_select_policy_path='default')
        instance = next(instances)
        observation, action_set, reward, done, _ = env.reset(instance)
        node_indices.append(env.active_node.node_index)
        observation_list.append(observation)

        while not done and reward > -5:
            if np.random.rand(1) > expert_prob:
                action_id = expert_policy.select_variable(observation, action_set)
            else:
                action_id = random_policy.select_variable(observation, action_set)

            observation, action_set, reward, done, _ = env.step(action_id)

            node_indices.append(env.active_node.node_index)
            observation_list.append(observation)
            
        for node in env.all_nodes:
            if node.optimal:
                found = False
                for i in range(len(node_indices)):
                    if node_indices[i] == node.node_index:
                        data = [observation_list[i], True, instance, env.w_opt, episode_counter]
                        found = True
                        break
                if not found:
                    continue
            else:
                found = False
                for i in range(len(node_indices)):
                    if node_indices[i] == node.node_index:
                        data = [observation_list[i], False, instance, env.w_opt, episode_counter]
                        found = True
                        break
                if not found:
                    continue
                    
            if collect_positive:
                if not max_samples_reached:
                    if node.optimal:
                        filename = Positive_sample_path + f'sample_{sample_counter}.pkl'
                        sample_counter += 1
            else:
                if not max_samples_reached:
                    # first = data[0].observation[:5]
                    # lb = data[0].observation[24+3*8*4+2:126]
                    # ub = data[0].observation[126:130]
                    # lb = data[0].variable_features[:,2]
                    # ub = data[0].variable_features[:,3]
                    
                    # print(lb, ub)
                    if node.optimal:
                        sample_counter += 1
                        filename = Positive_sample_path + f'sample_{sample_counter}.pkl'
                    else:
                        negative_counter += 1
                        filename = Negative_sample_path + f'sample_{negative_counter}.pkl'
                
                # else:
                #     filename = f'negative_node_samples2/sample_{sample_counter}.pkl'
                    
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)
                # If we collected enough samples, we finish the current episode but stop saving samples
                # if sample_counter >= max_samples:
                #     max_samples_reached = True
                #     break
        if episode_counter >= max_episodes:
            break
        print(f"Episode {episode_counter}, {sample_counter}, {negative_counter} samples collected so far")


if __name__=='__main__':
    collect_node_samples(observation_function=LinearObservation)
