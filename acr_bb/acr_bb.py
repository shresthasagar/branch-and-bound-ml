import ecole 
import torch
import numpy as np
from qp_relaxation_multicast import solve_qp_relaxed


class Observation(object):
    def __init__(self):
        self.antenna_features  = None # np.zeros(N, 3) # three features for each antenna
        self.variable_features = None # np.zeros(M, 15)
        self.edge_index = None
        self.edge_features     = None # np.zeros(N*M, 3)
        self.candidates        = None # np.arange(M)
        
        pass

    def extract(self, model):
        # TODO: make the observation out of the model 
        self.candidates = model.action_set_indices
        self.antenna_features = np.concatenate((model.active_node.w_sol.real, model.active_node.w_sol.imag, abs(model.active_node.w_sol)), axis=1)
        
        # edge features
        self.edge_indices = np.stack((np.repeat(np.arange(model.N), model.M), np.tile(np.arange(model.M), model.N)))
        self.edge_features = np.zeros((model.M*model.N, 3))
        self.edge_features[:,0] = np.real(model.H_complex.reshape(-1))
        self.edge_features[:,1] = np.imag(model.H_complex.reshape(-1))
        self.edge_features[:,2] = np.abs(model.H_complex.reshape(-1))

        # construct variable features
        # global features
        self.variable_features = np.zeros((model.M, 9))
        self.variable_features[:,0] = model.global_L # global lower bound
        self.variable_features[:,1] = model.global_U # global upper bound
        self.variable_features[:,2] = model.active_node.lb # constraint lower bound of the selected node
        self.variable_features[:,3] = model.active_node.ub # constraint upper bound of the selected node
        self.variable_features[:,4] = (model.active_node.U - model.global_U) < model.epsilon 

        # local features
        H_w = np.matmul(model.H_complex.conj().T, model.active_node.w_sol)
        self.variable_features[:,5] = np.squeeze(np.real(H_w))
        self.variable_features[:,6] = np.squeeze(np.imag(H_w))
        self.variable_features[:,7] = np.squeeze(np.abs(H_w))
        self.variable_features[:,8] = (np.squeeze(np.abs(H_w))>=1)*1

        #TODO: include the normalized number of times a variable has been selected by the current branching policy    
        return self

class Node(object):
    def __init__(self, lb=None, ub=None, w_sol=None, w_feas=None, U=False, L=False):
        assert lb is not None and ub is not None, " Cannot pass None to lb and ub"
        self.lb = lb
        self.ub = ub
        self.w_sol = w_sol
        self.w_feas = w_feas
        self.U = U
        self.L = L

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
    def __init__(self, observation_function=Observation(), epsilon=0.0001):
        self.observation_function = observation_function
        self._is_reset = None
        self.epsilon = epsilon # stopping criterion 
        self.H = None
        
        self.nodes = []     # list of problems (nodes)
        
        self.L_list = []    # list of lower bounds on the problem
        self.U_list = []    # list of upper bounds on the problem
        
        self.global_L = np.nan # global lower bound
        self.global_U = np.nan  # global upper bound        

        self.action_set_indices = None 

        # current active node
        self.active_node = None

        self.global_U_ind = None
    
    
    def is_terminal(self):
        if (self.global_U - self.global_L)/abs(self.global_U) < self.epsilon:
            return True
        else:
            return False

    def reset(self, instance):
        
        # clear all variables
        self.H = None
        self.nodes = []     # list of problems (nodes)
        self.L_list = []    # list of lower bounds on the problem
        self.U_list = []    # list of upper bounds on the problem
        self.global_L = np.nan # global lower bound
        self.global_U = np.nan  # global upper bound        
        self.action_set_indices = None 
        self.active_node = None
        self.global_U_ind = None


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
        [w, lower_bound, optimal] = solve_qp_relaxed(self.H, l , u)

        self.global_L = lower_bound
        self.w_opt = w/min(abs(np.matmul(self.H_complex.conj().T, w)))
 
        self.global_U = np.linalg.norm(self.w_opt)**2
        self.active_node = Node(lb=l, ub=u, w_sol=w, w_feas=self.w_opt, U=self.global_U, L=lower_bound) 
        
        done = False
        if self.is_terminal():
            done = True
        reward = 0
        observation = self.observation_function.extract(self)

        return observation, self.action_set_indices, reward, done, None

    def step(self, action_id):
        assert self._is_reset, "The environment instance has not been initialized"
        
        # use action_id branching variable to split its lower and upper bounds into two
        # construct child nodes 
        mid_ub = self.active_node.ub.copy()
        mid_ub[action_id] = (mid_ub[action_id] + self.active_node.lb[action_id])/2

        mid_lb = self.active_node.lb.copy()
        mid_lb[action_id] = mid_ub[action_id]


        # solve relaxed problem for the left child        
        [w_left, L_left, optimal] = solve_qp_relaxed(self.H, l=self.active_node.lb, u=mid_ub)
        w_feas = w_left/min(abs(np.matmul(self.H_complex.conj().T, w_left)))
        U_left = np.linalg.norm(w_feas)**2

        if self.global_U > U_left:
            self.global_U = U_left
            self.w_opt = w_feas 

        self.L_list.append(L_left)
        self.U_list.append(U_left)
        child_left = Node(lb=self.active_node.lb, ub=mid_ub, w_sol=w_left, w_feas=w_feas, U=U_left, L=L_left)
        

        # solve relaxed problem for the right child        
        [w_right, L_right, optimal] = solve_qp_relaxed(self.H, l=mid_lb, u=self.active_node.ub)
        w_feas = w_right/min(abs(np.matmul(self.H_complex.conj().T, w_right)))
        U_right = np.linalg.norm(w_feas)**2
        
        if self.global_U > U_right:
            self.global_U = U_right
            self.w_opt = w_feas 

        self.L_list.append(L_right)
        self.U_list.append(U_right)
        child_right = Node(lb=mid_lb, ub=self.active_node.ub, w_sol=w_right, w_feas=w_feas, U=U_right, L=L_right)

        self.nodes.append(child_left)
        self.nodes.append(child_right)

        min_ind = np.argmin([self.global_L, L_left, L_right])
        if min_ind == 1:
            self.global_L = L_left
            self.active_node = child_left
            self.active_node_index = len(self.nodes)-2
        elif min_ind == 2:
            self.global_L = L_right
            self.active_node = child_right
            self.active_node_index = len(self.nodes)-1
        else:
            self.active_node_index = np.argmin(self.L_list)
            self.active_node = self.nodes[self.active_node_index]        

        # Prune BB tree using the current bound

        # TODO: remove node selected
        del self.nodes[self.active_node_index]
        del self.L_list[self.active_node_index]
        del self.U_list[self.active_node_index]
        self.global_L = np.min(self.L_list)
        self.global_U = np.min(self.U_list)
        # Build observation, action_set is static, and reward
        observation = self.observation_function.extract(self) 
        reward = -1
        done = False
        # check stopping condition and assign to done
        if self.is_terminal():
            done = True

        return observation, self.action_set_indices, reward, done, None


if __name__ == '__main__':
    instance = np.random.randn(2,8,4)
    # H = np.array([[[ 1.27109919,  0.62238554],
    #         [-0.38933997,  0.48843181],
    #         [-0.14073963, -0.77918651]],

    #         [[-1.40115267, -0.17910412],
    #         [-0.82520195, -1.08825745],
    #         [-0.62317508, -0.67931762]]])
    env = ACRBBenv()
    obs, actions, reward, done, _  = env.reset(instance)
    if done:
        print('done')
        exit(0)
    policy = DefaultBranchingPolicy()
    for i in range(1000):
        action_id = policy.select_variable(obs, actions)
        obs, actions, reward, done, _ = env.step(action_id)
        if done:
            print('done')
            break
    print(done, i)
