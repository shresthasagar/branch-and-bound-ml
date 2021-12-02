import torch
import numpy as np
from single_beamforming.qp_relaxation_multicast import solve_qp_relaxed
from models.gnn_policy import GNNPolicy, GNNNodeSelectionPolicy
from models.gnn_dataset import get_graph_from_obs
from single_beamforming.fpp_sca import fpp_sca
from single_beamforming.observation import Observation, LinearObservation
import time 
from models.fcn_policy import FCNNodeSelectionLinearPolicy

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

class ACRBBenv(object):
    def __init__(self, observation_function=Observation, epsilon=0.0009, node_select_policy_path='default', policy_type='gnn', prune=True, init_U = None):
        """
        @params: 
            node_select_policy_path: one of {'default', 'oracle', policy_params}
                                     if the value is 'oracle', optimal solution should be provided in the reset function
                                     policy_params refers to the actual state_dict of the policy network
                                     appropriate policy_type should be given according the policy parameters provided in this argument
            policy_type: One of 'gnn' or 'linear'
        """
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
            if policy_type=='gnn':
                self.node_select_model = GNNPolicy()
                self.node_select_model.load_state_dict(torch.load(node_select_policy_path))
                self.node_select_policy = self.learnt_node_select
            if policy_type=='linear':
                self.node_select_model = FCNNodeSelectionLinearPolicy()
                self.node_select_model.load_state_dict(torch.load(node_select_policy_path))
                self.node_select_policy = self.learnt_node_select
                
        self.w_opt = None
        self.current_opt_node = None

        self.sigmoid = torch.nn.Sigmoid()
        self.min_bound_gap = None

        self.prune_policy = GNNNodeSelectionPolicy()

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
        self.min_bound_gap = np.ones(self.H.shape[-1])*0.01
        
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
            # print(oracle_opt.shape)
            self.optimal_angle = np.angle(np.matmul(self.H_complex.conj().T, self.oracle_opt))
            self.optimal_angle[self.optimal_angle<0] += 2*np.pi
            # print(self.optimal_angle)
        else:
            self.optimal_angle = np.random.randn(self.M, 1)

        return observation, self.action_set_indices, reward, done, None
          

    def push_children(self, action_id, node_id):
        self.delete_node(node_id)
        
        # use action_id branching variable to split its lower and upper bounds into two
        # construct child nodes 

        if not np.all(abs(self.active_node.ub - self.active_node.lb)> self.min_bound_gap):
            return

        mid_ub = self.active_node.ub.copy()
        mid_ub[action_id] = (mid_ub[action_id] + self.active_node.lb[action_id])/2

        mid_lb = self.active_node.lb.copy()
        mid_lb[action_id] = mid_ub[action_id]


        # solve relaxed problem for the left child        
        try:
            [w_left, L_left, optimal] = solve_qp_relaxed(self.H, l=self.active_node.lb, u=mid_ub)
        except:
            print(self.active_node.lb, self.active_node.ub)
            return True        

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
            return True

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

        self.num_nodes += 2
        self.num_active_nodes += 1

        self.global_L = np.min(self.L_list)
        # self.global_U = min([np.min(self.U_list), self.global_U])


    def set_node_select_policy(self, node_select_policy_path='default', policy_type='gnn'):
        if node_select_policy_path=='default':
            self.node_select_policy = 'default'
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = 'oracle'
        else:
            if policy_type == 'gnn':
                self.node_select_model = GNNNodeSelectionPolicy()
                self.node_select_model.load_state_dict(node_select_policy_path.state_dict())
                self.node_select_policy = 'ml_model'
            elif policy_type == 'linear':
                self.node_select_model = FCNNodeSelectionLinearPolicy()
                self.node_select_model.load_state_dict(node_select_policy_path.state_dict())
                self.node_select_policy = 'ml_model'
          
    def select_node(self):
        node_id = 0
        while len(self.nodes)>0:
            node_id = self.rank_nodes()
            fathomed = self.fathom(node_id)
            if fathomed:
                continue
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
            out = self.node_select_model(observation, 1)
            if out < 0.5:
                # print('prune')
                return True
            else:
                # print('select')
                return False

    def rank_nodes(self):
        return np.argmin(self.L_list)

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
        if (self.optimal_angle.squeeze()<=node.ub).all() and (self.optimal_angle.squeeze()>=node.lb).all():
            return True
        else:
            return False

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

        # Prune BB tree using the current bound
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
        self.active_node_index = np.argmin(self.L_list)
        self.active_node = self.nodes[self.active_node_index]        

        self.num_nodes += 2
        self.num_active_nodes += 1

        self.global_L = np.min(self.L_list)
        # self.global_U = min([np.min(self.U_list), self.init_U])

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


def solve_bb(instance, max_iter=1000):
    env = ACRBBenv(node_select_policy_path='default')
    obs, actions, reward, done, _  = env.reset(instance)
    if done:
        print('done')
        exit(0)
    policy = DefaultBranchingPolicy()
    for i in range(max_iter):
        print(i, env.global_U, env.global_L)
        action_id = policy.select_variable(obs, actions)
        obs, actions, reward, done, _ = env.step(action_id)
        if done:
            break
        # print(done, i, env.global_L, env.global_U, np.linalg.norm(env.w_opt,'fro')**2)
    # print(i)
    return env.w_opt.copy(), i

def solve_bb_policy(instance, max_iter=1000, policy='default', policy_type='gnn'):
    if policy_type == 'default':
        env = ACRBBenv(observation_function=Observation, epsilon=0.001)
    elif policy_type == 'gnn':
        env = ACRBBenv(observation_function=Observation, epsilon=0.001)
    elif policy_type == 'linear':
        env = ACRBBenv(observation_function=LinearObservation, epsilon=0.001)
    elif policy_type == 'oracle':
        env = ACRBBenv(observation_function=Observation, epsilon=0.001)
        pass
    branching_policy=DefaultBranchingPolicy()

    t1 = time.time()

    env.set_node_select_policy(node_select_policy_path=policy, policy_type=policy_type)
    env.reset(instance)
    timestep = 0
    done = False
    while timestep < max_iter and len(env.nodes)>0 and not done:
        node_id, node_feats, label = env.select_node()
        if len(env.nodes) == 0:
            break
        prune_node = env.prune(node_feats)
        if prune_node:
            env.delete_node(node_id)
            continue
        else:
            branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
            done = env.push_children(branching_var, node_id)
        timestep = timestep+1
    # returns the solution, objective value, timestep and the time taken
    return env.w_opt.copy(), np.linalg.norm(env.w_opt, 'fro')**2, timestep, time.time() - t1

if __name__ == '__main__':
    times = []
    num_eg = 1
    nodes_fpp = 0
    nodes_std = 0
    import scipy.io as sio
    H_mat = sio.loadmat('data/instance.mat')
    H = H_mat['H']
    for p in range(num_eg):
        print(p)
        # instance = np.random.randn(2,8,8)
        instance = np.stack((np.real(H), np.imag(H)), axis=0)
        # H = instance[0,:,:] + 1j*instance[1,:,:]

        # w = fpp_sca(H)
        # H = np.array([[[ 1.27109919,  0.62238554],
        #         [-0.38933997,  0.48843181],
        #         [-0.14073963, -0.77918651]],

        #         [[-1.40115267, -0.17910412],
        #         [-0.82520195, -1.08825745],
        #         [-0.62317508, -0.67931762]]])
        # env = ACRBBenv(node_select_policy_path='default', prune=True, init_U = np.linalg.norm(w)**2)
        w_opt, _ = solve_bb(instance, max_iter = 7000)
        print('here')
        print(np.linalg.norm(w_opt))
        print(w_opt)
