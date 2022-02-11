import torch
import torch.nn as nn
import numpy as np
import time 
from antenna_selection.beamforming import *
from antenna_selection.observation import Observation, LinearObservation
from models.gnn_policy import GNNPolicy, GNNNodeSelectionPolicy
from models.fcn_policy import FCNNodeSelectionLinearPolicy
from models.gnn_dataset import get_graph_from_obs
from models.setting import MODEL_PATH
from beamforming_test import Beamforming, BeamformingWithSelectedAntennas

class Node(object):
    def __init__(self, z_mask=None, z_sol=None, z_feas=None, W_sol=None, U=False, L=False, depth=0, parent_node=None, node_index = 0):
        """
        @params: 
            z_mask: vector of boolean, True means that the current variable is boolean
            z_sol: value of z at the solution of the cvx relaxation
            z_feas: value of z after making z_sol feasible (i.e. boolean)
            U: True if it is the current global upper bound
            L: True if it is the current global lower bound
            depth: depth of the node from the root of the BB tree
            node_index: unique index assigned to the node in the BB tree
        """
        self.z_mask = z_mask.copy()
        self.z_sol = z_sol.copy()
        self.z_feas = z_feas.copy()
        self.W_sol = W_sol.copy()
        self.U = U
        self.L = L
        self.depth = depth
        self.parent_node = parent_node
        self.node_index = node_index

class DefaultBranchingPolicy(object):
    def __init__(self):
        pass

    def select_variable(self, observation, candidates):
        z_mask = observation.antenna_features[:, 2]
        z_sol = observation.antenna_features[:,0]
        z_sol_rel = (1-z_mask)*(np.abs(z_sol - 0.5))
        return np.argmax(z_sol_rel)


class ASBBenv(object):
    def __init__(self, observation_function=Observation, node_select_policy_path='default', policy_type='gnn', epsilon=0.001):
        """
        @params: 
            node_select_policy_path: one of {'default', 'oracle', policy_params}
                                     if the value is 'oracle', optimal solution should be provided in the reset function
                                     policy_params refers to the actual state_dict of the policy network
                                     appropriate policy_type should be given according the policy parameters provided in this argument
            policy_type: One of 'gnn' or 'linear'
        """
        self._is_reset = None
        self.epsilon = epsilon # stopping criterion 
        self.H = None
        
        self.nodes = []     # list of problems (nodes)
        self.num_nodes = 0
        self.num_active_nodes = 0
        self.all_nodes = [] # list of all nodes to serve as training data for node selection policy
        self.optimal_nodes = []
        self.node_index_count = 0

        self.L_list = []    # list of lower bounds on the problem
        self.U_list = []    # list of upper bounds on the problem
        
        self.global_L = np.nan # global lower bound
        self.global_U = np.nan  # global upper bound        

        self.action_set_indices = None 
        # current active node
        self.active_node = None

        self.global_U_ind = None
        self.failed_reward = -2000

        self.node_select_model = None

        self.init_U = 999999
        self.node_select_policy = self.default_node_select        
        
        self.z_incumbent = None
        self.W_incumbent = None
        self.current_opt_node = None
        self.min_bound_gap = None

        if node_select_policy_path == 'default':
            self.node_select_policy = self.default_node_select
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = self.oracle_node_select
        else:
            if policy_type=='gnn':
                self.node_select_model = GNNPolicy()
                self.node_select_model.load_state_dict(torch.load(node_select_policy_path))
                self.node_select_policy = self.learnt_node_select
            if policy_type=='linear':
                self.node_select_model = FCNNodeSelectionLinearPolicy()
                self.node_select_model.load_state_dict(torch.load(node_select_policy_path))
                self.node_select_policy = self.learnt_node_select
                
        self.observation_function = observation_function
        self.include_heuristic_solutions = False
        self.heuristic_solutions = []
        
        self.bm_solver = None

    def set_heuristic_solutions(self, solution):
        """
        Provide antenna selections provided by heuristic methods in order to incorporate them into the BB
        """
        self.include_heuristic_solutions = True
        self.heuristic_solutions.append(solution)


    def reset(self, instance, max_ant,  oracle_opt=None):
        
        # clear all variables
        self.H = None
        self.nodes = []  # list of problems (nodes)
        self.all_nodes = []
        self.optimal_nodes = []
        self.node_index_count = 0

        self.L_list = []    # list of lower bounds on the problem
        self.U_list = []    # list of upper bounds on the problem
        self.global_L = np.nan # global lower bound
        self.global_U = np.nan  # global upper bound        
        self.action_set_indices = None 
        self.active_node = None
        self.global_U_ind = None
        self.num_nodes = 1

        self.H = instance
        self.H_complex = self.H[0,:,:] + self.H[1,:,:]*1j
        self.bm = Beamforming(self.H_complex, max_ant=max_ant)
        self.bm_with_antennas = BeamformingWithSelectedAntennas(self.H_complex, max_ant=max_ant)

        self.min_bound_gap = np.ones(self.H.shape[-1])*0.01
        
        self.max_ant = max_ant

        # number of transmitters and users
        _, self.N, self.M = self.H.shape 
        self._is_reset = True
        self.action_set_indices = np.arange(1,self.N)

        # boolean vars corresponding to each antenna denoting its selection if True
        z_mask = np.zeros(self.N)
        # values of z (selection var) at the z_mask locations
        # for the root node it does not matter
        z_sol = np.zeros(self.N)

        done = False
        # initialize the root node 
        # try:
        [z, W, lower_bound] = solve_beamforming_relaxed(self.H_complex, max_ant=self.max_ant, z_mask=z_mask, z_sol=z_sol)
        # except:
        #     print("exeption occured")
        #     return None, self.action_set_indices, -1000, done, None

        self.global_L = lower_bound
        # print('reset z, H', z, self.H_complex.shape)
        self.z_incumbent = self.get_feasible_z(z)
        W_feas, self.global_U = solve_beamforming_with_selected_antennas(self.H_complex, self.z_incumbent)

        if not self.global_U == np.inf:
            self.W_incumbent = W_feas.copy()
        else:
            self.W_incumbent = np.zeros(self.H_complex.shape)

        self.active_node = Node(z_mask=z_mask, z_sol=z, z_feas=self.z_incumbent, W_sol = W, U=self.global_U, L=lower_bound, depth=1, node_index=self.node_index_count) 
        self.current_opt_node = self.active_node
        
        self.active_node_index = 0
        self.nodes.append(self.active_node)
        self.L_list.append(lower_bound)
        self.U_list.append(self.global_U)
        self.all_nodes.append(self.active_node)

        done = False
        if self.is_terminal():
            done = True
        reward = 0

        # TODO: build observation function
        observation = self.observation_function().extract(self)
        # observation = None

        if oracle_opt is not None:
            self.oracle_opt = oracle_opt
        else:
            self.oracle_opt = np.zeros(self.N)

        return 

    def push_children(self, action_id, node_id):
        self.delete_node(node_id)
        
        # use action_id branching variable to split its lower and upper bounds into two
        # construct child nodes 


        z_mask_left = self.active_node.z_mask.copy()
        z_mask_left[action_id] = 1

        z_mask_right = self.active_node.z_mask.copy()
        z_mask_right[action_id] = 1

        z_sol_left = self.active_node.z_sol.copy()
        z_sol_left[action_id] = 0

        z_sol_right = self.active_node.z_sol.copy()
        z_sol_right[action_id] = 1
        
        min_L_child = self.global_L

        t = 0;  
        # check if the current problem has only one feasible solution (do not construct new node in that case)
        if not np.sum(np.round(z_sol_left)*z_mask_left)>=self.max_ant:
            # solve the relaxed formulation
            t1 = time.time()
            print('z mask', z_mask_left)
            print('z sol', z_sol_left)
            [z_left, W_left, L_left] = self.bm.solve_beamforming(z_mask=z_mask_left, z_sol=z_sol_left)
            print('solution', L_left)
            
            # [z_left, W_left, L_left] = solve_beamforming_relaxed(self.H_complex, max_ant=self.max_ant, z_mask=z_mask_left, z_sol=z_sol_left)
            t += time.time() - t1
            assert L_left >= self.active_node.L - self.epsilon, 'lower bound of left child node less than that of parent'

            if not L_left == np.inf:
                # check if any of the solution variables are boolean
                temp = (1-z_mask_left)*(np.abs(z_left - 0.5))
                z_mask_left[temp>0.499] = 1

                z_feas_left = self.get_feasible_z(z_left)
                
                t1 = time.time()
                W_feas, U_left = self.bm_with_antennas.solve_beamforming(z=z_feas_left)

                # W_feas, U_left = solve_beamforming_with_selected_antennas(self.H_complex, z_feas_left)
                t += time.time() - t1

                # only append the node if it is not yet feasible
                if not np.sum(np.round(z_left)*z_mask_left)>=self.max_ant:
                    self.node_index_count += 1
                    child_left = Node(z_mask=z_mask_left, z_sol=z_left, z_feas=z_feas_left, W_sol=W_left, U=U_left, L=L_left, depth=self.active_node.depth+1, node_index=self.node_index_count) 
                    self.L_list.append(L_left)
                    self.U_list.append(U_left)
                    self.nodes.append(child_left)
                    self.all_nodes.append(child_left)
            else:
                U_left = np.inf
        else:
            t1 = time.time()
            W_feas, U_left = self.bm_with_antennas.solve_beamforming(z = np.round(z_sol_left)*z_mask_left)
            
            # W_feas, U_left = solve_beamforming_with_selected_antennas(self.H_complex, np.round(z_sol_left)*z_mask_left)

            t += time.time() - t1

            L_left = U_left
            z_feas_left = np.round(z_sol_left)*z_mask_left

        # solve relaxed problem for the left child        

        if self.global_U > U_left:
            self.global_U = U_left
            self.z_incumbent = z_feas_left.copy()
            self.W_incumbent = W_feas.copy()
            # self.current_opt_node = child_left

        

        if not np.sum(np.round(z_sol_right)*z_mask_right)>=self.max_ant:
            # solve relaxed problem for the right child        
            
            t1 = time.time()
            
            [z_right, W_right, L_right] = self.bm.solve_beamforming(z_mask=z_mask_right, z_sol=z_sol_right)
            # [z_right, W_right, L_right] = solve_beamforming_relaxed(self.H_complex, max_ant=self.max_ant, z_mask=z_mask_right, z_sol=z_sol_right)
            t += time.time() - t1
            assert L_right >= self.active_node.L - self.epsilon, 'lower bound of right child node less than that of parent'
            
            if not L_right == np.inf:
                temp = (1-z_mask_right)*(np.abs(z_right - 0.5))
                z_mask_right[temp>0.499] = 1

                z_feas_right = self.get_feasible_z(z_right)
                
                t1 = time.time()
                W_feas, U_right = self.bm_with_antennas.solve_beamforming(z=z_feas_right)

                # W_feas, U_right = solve_beamforming_with_selected_antennas(self.H_complex, z_feas_right)
                t += time.time() - t1

                if not np.sum(np.round(z_right)*z_mask_right)>=self.max_ant:
                    self.node_index_count += 1
                    child_right = Node(z_mask=z_mask_right, z_sol=z_right, z_feas=z_feas_right, W_sol=W_right, U=U_right, L=L_right, depth=self.active_node.depth+1, node_index=self.node_index_count) 
                    self.L_list.append(L_right)
                    self.U_list.append(U_right)
                    self.nodes.append(child_right)
                    self.all_nodes.append(child_right)
            else:
                U_left = np.inf
        else:
            
            t1 = time.time()
            W_feas, U_right = self.bm_with_antennas.solve_beamforming(z=np.round(z_sol_right)*z_mask_right)

            # W_feas, U_right = solve_beamforming_with_selected_antennas(self.H_complex, np.round(z_sol_right)*z_mask_right)
            t += time.time() - t1
            L_right = U_right
            z_feas_right = np.round(z_sol_right)*z_mask_right

        if self.global_U > U_right:
            self.global_U = U_right
            self.z_incumbent = z_feas_right.copy()
            self.W_incumbent = W_feas.copy()
            # self.current_opt_node = child_right

        if len(self.nodes) == 0:
            return
        min_L_child = min(L_left, L_right)
        self.global_L = min(min(self.L_list), min_L_child)
        print("TOTAL TIME cvx", t)
            
        # self.global_U = min([np.min(self.U_list), self.global_U])


    def set_node_select_policy(self, node_select_policy_path='default', policy_type='gnn'):
        if node_select_policy_path=='default':
            self.node_select_policy = 'default'
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = 'oracle'
        else:
            if policy_type == 'gnn':
                self.node_select_model = GNNNodeSelectionPolicy()
                # self.node_select_model.load_state_dict(node_select_policy_path.state_dict())
                print('policy path', node_select_policy_path)
                model_state_dict = torch.load(node_select_policy_path)
                self.node_select_model.load_state_dict(model_state_dict)
                self.node_select_policy = 'ml_model'

            elif policy_type == 'linear':
                self.node_select_model = FCNNodeSelectionLinearPolicy()
                # self.node_select_model.load_state_dict(node_select_policy_path.state_dict())
                self.node_select_model.load_state_dict(node_select_policy_path)
                self.node_select_policy = 'ml_model'

    def select_variable_default(self):
        z_sol_rel = (1-self.active_node.z_mask)*(np.abs(self.active_node.z_sol - 0.5))
        return np.argmax(z_sol_rel)



    def select_node(self):
        node_id = 0
        while len(self.nodes)>0:
            node_id = self.rank_nodes()
            # fathomed = self.fathom(node_id)
            # if fathomed:
            #     print('fathomed')
            #     continue
            self.active_node = self.nodes[node_id]
            break
        return node_id, self.observation_function().extract(self), self.is_optimal(self.active_node)


    def prune(self, observation):
        if isinstance(observation, Observation):
            observation = get_graph_from_obs(observation, self.action_set_indices)
        elif isinstance(observation, LinearObservation):
            observation = torch.tensor(observation.observation, dtype=torch.float32).unsqueeze(0)
        if self.node_select_policy == 'oracle':
            return not self.is_optimal(self.active_node)
        elif self.node_select_policy == 'default':
            return False
        else:
            # out = self.node_select_model(observation.antenna_features, observation.edge_index, observation.edge_attr, observation.variable_features) 
            # out = out.sum()
            # out = self.sigmoid(out) 
            if self.include_heuristic_solutions:
                heuristic_match = self.contains_heuristic(self.active_node)
                if heuristic_match:
                    return False

            with torch.no_grad():
                out = self.node_select_model(observation, 1)

            if out < 0.5:
                # print('prune')
                return True
            else:
                # print('select')
                return False

    def rank_nodes(self):
        return np.argmin(self.L_list)

    def fathom_nodes(self):
        del_ind = np.argwhere(np.array(self.L_list) > self.global_U)
        if len(del_ind)>0:
            del_ind = sorted(list(del_ind.squeeze(axis=1)))
            for i in reversed(del_ind):
                self.delete_node(i)
        
    def fathom(self, node_id):
        if self.nodes[node_id].L > self.global_U:
            self.delete_node(node_id)
            return True
        return False

    def delete_node(self, node_id):
        del self.nodes[node_id]
        del self.L_list[node_id]
        del self.U_list[node_id]

    def is_optimal(self, node):
        if np.linalg.norm(node.z_mask*(node.z_sol - self.oracle_opt)) < 0.0001:
            return True
        else:
            return False

    def contains_heuristic(self, node):
        contains = False
        for heuristic_sol in self.heuristic_solutions:
            if np.linalg.norm(node.z_mask*(node.z_sol - heuristic_sol)) < 0.0001:
                contains = True
                break
        return contains

    def is_terminal(self):
        if (self.global_U - self.global_L)/abs(self.global_U) < self.epsilon:
            return True
        else:
            return False

    def default_node_select(self):
        """
        Use the node with the lowest lower bound
        """
        return np.argmin(self.L_list)

    def get_feasible_z(self, z):
        z_round = np.round(z)
        if np.sum(z_round) <= self.max_ant:
            return z_round
        else:
            mask = np.zeros(len(z))
            mask[np.argsort(z)[len(z)-self.max_ant:]] = 1
            return mask



def solve_bb(instance, max_ant=5, max_iter=1000, policy='default', policy_type='gnn', oracle_opt=None):
    t1 = time.time()
    if policy_type == 'default':
        env = ASBBenv(observation_function=Observation, epsilon=0.001)
    elif policy_type == 'gnn':
        env = ASBBenv(observation_function=Observation, epsilon=0.001)
    elif policy_type == 'linear':
        env = ASBBenv(observation_function=LinearObservation, epsilon=0.001)
    elif policy_type == 'oracle':
        env = ASBBenv(observation_function=Observation, epsilon=0.001)
        pass

    branching_policy = DefaultBranchingPolicy()

    t1 = time.time()

    env.reset(instance, max_ant=max_ant)
    timestep = 0
    done = False
    print("INITIALIZATION COMPLETED: ", time.time()-t1)
    # print(len(env.nodes))
    while timestep < max_iter and len(env.nodes)>0 and not done:
        print('timestep', timestep, env.global_U, env.global_L)
        # t1 = time.time()
        env.fathom_nodes()
        if len(env.nodes) == 0:
            break
        node_id, node_feats, label = env.select_node()
        
        if len(env.nodes) == 0:
            break
        # prune_node = env.prune(node_feats)
        # if prune_node:
        #     env.delete_node(node_id)
        #     continue
        # else:
        branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
        # print("NOW BRANCHING: ", time.time()-t1)
        t1 = time.time()
        done = env.push_children(branching_var, node_id)
        timestep = timestep+1
        print("BRANCHING COMPLETED: ", time.time()-t1)

    print('ended')
    # returns the solution, objective value, timestep and the time taken
    return env.z_incumbent.copy(), env.global_U, timestep , time.time()-t1

if __name__ == '__main__':
    np.random.seed(seed = 100)
    N = 12
    M = 6
    max_ant = 5
    
    u_avg = 0
    t_avg = 0
    tstep_avg = 0
    for i in range(1):
        H = np.random.randn(N, M) + 1j*np.random.randn(N,M)    
        instance = np.stack((np.real(H), np.imag(H)), axis=0)
        global_U, _, timesteps, t = solve_bb(instance, max_ant=max_ant, max_iter = 7000)
        u_avg += global_U
        t_avg += t
        tstep_avg += timesteps

    print(u_avg, t_avg, tstep_avg)

    # print('bb solution: {}, optimal: {}'.format(global_U, optimal_f) )
