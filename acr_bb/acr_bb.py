import torch
import numpy as np
from qp_relaxation_multicast import solve_qp_relaxed
from gnn_policy import GNNPolicy
from helpers import get_graph_from_obs
from fpp_sca import fpp_sca
from observation import Observation, LinearObservation

class Node(object):
    def __init__(self, lb=None, ub=None, w_sol=None, w_feas=None, U=False, L=False, depth=0, parent_node=None, node_index = 0):
        assert lb is not None and ub is not None, " Cannot pass None to lb and ub"
        self.lb = lb
        self.ub = ub
        self.w_sol = w_sol
        self.w_feas = w_feas
        self.U = U
        self.L = L
        self.depth = depth
        self.parent_node = parent_node
        self.optimal = False
        self.node_index = node_index

class DefaultBranchingPolicy(object):
    def __init__(self):
        pass

    def select_variable(self, observation, candidates):
        mask = np.ones(observation.variable_features.shape[0])*1000000
        mask[candidates] = 0
        selection = np.argmin(observation.variable_features[:, 7]+ mask)
        return selection

class RandomPolicy(object):
    def __init__(self):
        pass

    def select_variable(self, observation, candidates):
        selection = np.random.choice(candidates)
        return selection

def instance_generator(M, N):
    while 1:
        yield np.random.randn(2,N,M)      

class ACRBBenv(object):
    def __init__(self, observation_function=Observation, epsilon=0.0001, node_select_policy_path='default', prune=True, init_U = None):
        self.observation_function = observation_function
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
        self.prune = prune
        # current active node
        self.active_node = None

        self.global_U_ind = None
        self.failed_reward = -2000

        self.node_select_model = None
        


        self.init_U = 999999
        if init_U is not None:
            self.init_U = init_U

        if node_select_policy_path == 'default':
            self.node_select_policy = self.default_node_select
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = self.oracle_node_select
        else:
            self.node_select_model = GNNPolicy()
            self.node_select_model.load_state_dict(torch.load(node_select_policy_path))
            self.node_select_policy = self.learnt_node_select
            
        self.w_opt = None
        self.current_opt_node = None

    def set_node_select_policy(self, node_select_policy_path='default'):
        if node_select_policy_path=='default':
            self.node_select_policy = self.default_node_select
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = self.oracle_node_select
        else:
            # self.node_select_model = GNNPolicy()
            # self.node_select_model.load_state_dict(torch.load(node_select_policy_path))
            self.node_select_model = node_select_policy_path
            self.node_select_policy = self.learnt_node_select


    def is_terminal(self):
        if (self.global_U - self.global_L)/abs(self.global_U) < self.epsilon:
            return True
        else:
            return False

    def reset(self, instance, oracle_opt=None):
        
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
        
        # number of transmitters and users
        _, self.N, self.M = self.H.shape 
        self._is_reset = True
        self.action_set_indices = np.arange(1,self.M)
        # initialize the lists with root node parameters
        # self.L_l

        l = np.zeros(self.M)
        u = np.ones(self.M)*2*np.pi

        # initialize the root node 
        try:
            [w, lower_bound, optimal] = solve_qp_relaxed(self.H, l , u)
        except:
            print(self.active_node.lb, self.active_node.ub)
            done = False
            return None, self.action_set_indices, -1000, done, None

        self.global_L = lower_bound
        self.w_opt = w/min(abs(np.matmul(self.H_complex.conj().T, w)))
        self.global_U = min([np.linalg.norm(self.w_opt)**2, self.init_U])
        self.active_node = Node(lb=l, ub=u, w_sol=w, w_feas=self.w_opt, U=self.global_U, L=lower_bound, depth=1, node_index=self.node_index_count) 
        self.current_opt_node = self.active_node
        
        self.active_node_index = 0
        self.nodes.append(self.active_node)
        self.L_list.append(lower_bound)
        self.U_list.append(self.global_U)

        done = False
        if self.is_terminal():
            done = True
        reward = 0
        observation = self.observation_function().extract(self)

        self.all_nodes.append(self.active_node)

        self.optimal_angle = None
        if oracle_opt is not None:
            self.oracle_opt = oracle_opt
            print(oracle_opt.shape)
            self.optimal_angle = np.angle(np.matmul(self.H_complex.conj().T, self.oracle_opt))
            self.optimal_angle[self.optimal_angle<0] += 2*np.pi
            print(self.optimal_angle)

        return observation, self.action_set_indices, reward, done, None

    def default_node_select(self):
        """
        Use the node with the lowest lower bound
        """
        return np.argmin(self.L_list)

    def is_optimal(self, node):
        if (self.optimal_angle.squeeze()<=node.ub).all() and (self.optimal_angle.squeeze()>=node.lb).all():
            return True
        else:
            return False

    def oracle_node_select(self):
        print(len(self.nodes))
        for i in range(len(self.nodes)):
            # print(self.nodes[i].lb, self.nodes[i].ub)
            if self.is_optimal(self.nodes[i]):
                self.nodes = [self.nodes[i]]
                return 0
        print('could not find optimal node in the list')
        exit(0) 

    def learnt_node_select(self, path='training/data/trained_params_gnn2.pkl'):
        """
        Use the learnt policy to select the node
        """
        import torch.nn as nn
        sigmoid = nn.Sigmoid()
        outs = []
        sort_ind = np.argsort(self.L_list)
        self.L_list.sort()
        self.U_list = [self.U_list[i] for i in sort_ind]
        self.nodes = [self.nodes[i] for i in sort_ind]
        if np.random.rand(1) > 0.0:
            # print('remaining nodes', len(self.nodes))
            for i in range(len(self.nodes)-1, -1, -1):
            # for counter in range(len(sort_ind)):
                self.active_node = self.nodes[i]  
                observation = self.observation_function().extract(self)
                observation = get_graph_from_obs(observation, self.action_set_indices)
                out = self.node_select_model(observation.antenna_features, observation.edge_index, observation.edge_attr, observation.variable_features) 
                out = out.sum()
                out = sigmoid(out)
                outs.append(out)
                if out > 0.5:
                    print('selected index is ', i, len(self.L_list))
                    # for node_ind in range(i):
                        # del self.nodes[0]
                        # del self.L_list[0]
                        # del self.U_list[0]
                    return i
        # return np.argmax(outs)
        return np.argmin(self.L_list)

    def step(self, action_id):
        assert self._is_reset, "The environment instance has not been initialized"
        
        # TODO: remove node selected
        del self.nodes[self.active_node_index]
        del self.L_list[self.active_node_index]
        del self.U_list[self.active_node_index]

        # use action_id branching variable to split its lower and upper bounds into two
        # construct child nodes 
        mid_ub = self.active_node.ub.copy()
        mid_ub[action_id] = (mid_ub[action_id] + self.active_node.lb[action_id])/2

        mid_lb = self.active_node.lb.copy()
        mid_lb[action_id] = mid_ub[action_id]


        # solve relaxed problem for the left child        
        try:
            [w_left, L_left, optimal] = solve_qp_relaxed(self.H, l=self.active_node.lb, u=mid_ub)
        except:
            print(self.active_node.lb, self.active_node.ub)
            done = False
            return self.observation_function().extract(self), self.action_set_indices, -1000, done, None
        
        w_feas = w_left/min(abs(np.matmul(self.H_complex.conj().T, w_left)))
        U_left = np.linalg.norm(w_feas)**2

        child_left = Node(lb=self.active_node.lb, ub=mid_ub, w_sol=w_left, w_feas=w_feas, U=U_left, L=L_left, depth=self.active_node.depth+1, parent_node=self.active_node, node_index = self.node_index_count)

        if self.global_U > U_left:
            self.global_U = U_left
            self.w_opt = w_feas.copy()
            self.current_opt_node = child_left

        self.L_list.append(L_left)
        self.U_list.append(U_left)
        self.node_index_count += 1
        

        # solve relaxed problem for the right child        
        try:
            [w_right, L_right, optimal] = solve_qp_relaxed(self.H, l=mid_lb, u=self.active_node.ub)
        except:
            print(self.active_node.lb, self.active_node.ub)
            done = False
            return self.observation_function().extract(self), self.action_set_indices, -1000, done, None
        
        w_feas = w_right/min(abs(np.matmul(self.H_complex.conj().T, w_right)))
        U_right = np.linalg.norm(w_feas)**2
        
        child_right = Node(lb=mid_lb, ub=self.active_node.ub, w_sol=w_right, w_feas=w_feas, U=U_right, L=L_right, depth=self.active_node.depth+1, parent_node=self.active_node, node_index = self.node_index_count)
        
        if self.global_U > U_right:
            self.global_U = U_right
            self.w_opt = w_feas.copy()
            self.current_opt_node = child_right

        self.L_list.append(L_right)
        self.U_list.append(U_right)
        self.node_index_count += 1

        self.nodes.append(child_left)
        self.nodes.append(child_right)
        self.all_nodes.append(child_left)
        self.all_nodes.append(child_right)

        # print('lower bounds', L_right, L_left)

        # min_ind = np.argmin([self.global_L, L_left, L_right])
        # if min_ind == 1:
        #     self.global_L = L_left
        #     self.active_node = child_left
        #     self.active_node_index = len(self.nodes)-2
        # elif min_ind == 2:
        #     self.global_L = L_right
        #     self.active_node = child_right
        #     self.active_node_index = len(self.nodes)-1

        # Prune BB tree using the current bound
        if self.prune:
            i = 0
            while i < len(self.nodes):
                if self.nodes[i].L > self.global_U:
                    del self.nodes[i]
                    del self.L_list[i]
                    del self.U_list[i]
                    # print('nodes pruned', len(self.nodes))
                else:
                    i +=1

        # Node selection
        # self.active_node_index = np.argmin(self.L_list)
        self.active_node_index = self.node_select_policy()
        self.active_node = self.nodes[self.active_node_index]        

        self.num_nodes += 2
        self.num_active_nodes += 1

        self.global_L = np.min(self.L_list)
        self.global_U = min([np.min(self.U_list), self.init_U])

        # Build observation, action_set is static, and reward
        observation = self.observation_function().extract(self) 
        reward = -1
        done = False
        # check stopping condition and assign to done
        if self.is_terminal():
            done = True
            self.optimal_nodes = [self.current_opt_node]
            self.optimal_nodes[-1].optimal = True
            while self.optimal_nodes[-1].parent_node is not None:
                self.optimal_nodes.append(self.optimal_nodes[-1].parent_node)
                self.optimal_nodes[-1].optimal = True
            # print('num optimal nodes', len(self.optimal_nodes))

        return observation, self.action_set_indices, reward, done, None


if __name__ == '__main__':
    times = []

    # instance = np.random.randn(2,4,8)
    # env = ACRBBenv(observation_function=LinearObservation)
    # obs, actions, reward, done, _  = env.reset(instance)
    # print(obs.observation.shape)

    num_eg = 1
    nodes_fpp = 0
    nodes_std = 0
    for p in range(num_eg):
        print(p)
        instance = np.random.randn(2,8,4)
        H = instance[0,:,:] + 1j*instance[1,:,:]

        # w = fpp_sca(H)
        # H = np.array([[[ 1.27109919,  0.62238554],
        #         [-0.38933997,  0.48843181],
        #         [-0.14073963, -0.77918651]],

        #         [[-1.40115267, -0.17910412],
        #         [-0.82520195, -1.08825745],
        #         [-0.62317508, -0.67931762]]])
        # env = ACRBBenv(node_select_policy_path='default', prune=True, init_U = np.linalg.norm(w)**2)
        env = ACRBBenv(node_select_policy_path='default')
        obs, actions, reward, done, _  = env.reset(instance)
        if done:
            print('done')
            exit(0)
        policy = DefaultBranchingPolicy()
        for i in range(1000):
            # print(i)
            action_id = policy.select_variable(obs, actions)
            obs, actions, reward, done, _ = env.step(action_id)
            if done:
                # print('done')
                nodes_fpp += i
                break
            print(done, i, p, env.global_L, env.global_U, np.linalg.norm(env.w_opt,'fro')**2)
        
        w_opt = env.w_opt.copy()

        env = ACRBBenv(node_select_policy_path='oracle', prune=True)
        obs, actions, reward, done, _  = env.reset(instance, oracle_opt=w_opt)
        if done:
            print('done')
            exit(0)
        policy = DefaultBranchingPolicy()
        for i in range(1000):
            # print(i)
            action_id = policy.select_variable(obs, actions)
            obs, actions, reward, done, _ = env.step(action_id)
            if done:
                # print('done')
                nodes_std += i
                break
            print(done, i, p, env.global_L, env.global_U, np.linalg.norm(env.w_opt,'fro')**2)

        # # print(env.global_L, np.linalg.norm(w)**2)
        # print(nodes_std/20.0, nodes_fpp/20.0)

        # env = ACRBBenv(node_select_policy_path='training/data/trained_params_gnn2.pkl')
        # obs, actions, reward, done, _  = env.reset(instance)
        # if done:
        #     print('done')
        #     exit(0)
        # policy = DefaultBranchingPolicy()
        # for i in range(1000):
        #     print(i)
        #     action_id = policy.select_variable(obs, actions)
        #     obs, actions, reward, done, _ = env.step(action_id)
        #     if done:
        #         print('done')
        #         break
        #     print(done, i, p, env.global_L, env.global_U, np.linalg.norm(env.w_opt,'fro')**2)
        # print(obs.antenna_features.shape)
        # print(obs.antenna_features)

    #     print(env.active_node.w_sol)
    #     # hw = np.matmul(env.H_complex.conj().T, env.w_opt)
    #     # angle = np.angle(hw)
    #     # angle[angle<0] += 2*np.pi
    #     # for i in range(len(env.optimal_nodes)):
    #     #     print('node',i)
    #     #     print(angle.squeeze())
    #     #     print(env.optimal_nodes[i].lb)
    #     #     print(env.optimal_nodes[i].ub)
    #     times.append(i)
    # print(np.mean(times))
