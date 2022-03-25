import torch
import torch.nn as nn
import numpy as np
import time 
from models.gnn_policy import GNNPolicy, GNNNodeSelectionPolicy
from models.fcn_policy import FCNNodeSelectionLinearPolicy
from models.gnn_dataset import get_graph_from_obs
from models.setting import MODEL_PATH, DEBUG
from single_group_as_bm.observation import *
from single_group_as_bm.solve_relaxation import qp_relaxed, qp_relaxed_with_selected_antennas, cvxpy_relaxed
import numpy.ma as ma

min_bound_gap = 0.01

class Node(object):
    def __init__(self,
                z_mask=None, 
                z_sol=None, 
                z_feas=None, 
                w_sol=None, 
                w_feas=None,
                l_angle=None,
                u_angle=None,
                U=False, 
                L=False, 
                depth=0, 
                parent_node=None, 
                node_index = 0):
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
        # if not self.check_integrality(z_sol, z_mask):
        #     print('assertion failed', z_sol, z_mask)
        assert self.check_integrality(z_sol, z_mask), "Creating Node: selected antennas part not integral"
        # related to the discrete variables
        self.z_mask = z_mask.copy()
        self.z_sol = z_sol.copy()
        self.z_feas = z_feas.copy()

        # related to the continuous variables
        self.w_sol = w_sol.copy()
        self.w_feas = w_feas.copy()
        self.l_angle = l_angle.copy()
        self.u_angle = u_angle.copy()

        # BB statistics
        self.U = U
        self.L = L
        self.depth = depth
        self.parent_node = parent_node
        self.node_index = node_index

    def check_integrality(self, z_sol, z_mask):
        sum_z= np.sum(np.abs(z_mask*(z_sol - np.round(z_sol))))
        return sum_z < 0.001

class DefaultBranchingPolicy(object):
    def __init__(self):
        pass

    def select_variable(self, observation, candidates, env=None):
        """
        Returns two variables to split on resulting in 4 children at one node
        """
        # z branching variable
        z_mask = observation.antenna_features[:,2]
        z_sol = observation.antenna_features[:,0]

        z_sol_rel = (1-z_mask)*(np.abs(z_sol - 0.5))
        if sum(1-z_mask)<1:
            z_branching_var = None
        else:
            # z_sol_rel[z_sol_rel==0]= 9999
            z_branching_var = np.argmax(z_sol_rel)

        # maximum channel engergy
        # if env is not None:
        #     z_compare = (1-z_mask)*env.H_energy_per_antenna
        #     if sum(1-z_mask)<1:
        #         z_branching_var = None
        #     else:
        #         # if np.random.rand() > 0.99:
        #             # max_candidate = np.argmax()
        #         z_branching_var = np.argmax(z_compare)
        #         # else:
        #         #     z_compare[z_compare==0]= 9999
        #         #     z_branching_var = np.argmin(z_compare)
                    
                       

        # maximum channel sum
        # if env is not None:
        #     z_compare = (1-z_mask)*env.H_sum_per_antenna
        #     if sum(1-z_mask)<1:
        #         z_branching_var = None
        #     else:
        #         z_branching_var = np.argmax(z_compare)


        # w branching variable
        mask = np.zeros(observation.variable_features.shape[0])
        mask[0] = 1 # To make the variable not= 1
        l_angle = observation.variable_features[:,3]
        u_angle = observation.variable_features[:,4]
        
        for i in range(1,len(mask)):
            if u_angle[i] - l_angle[i] < min_bound_gap:
                mask[i] = 1
        
        if sum(mask) < len(mask):
            w_branching_var = np.argmin(observation.variable_features[:, 7]+ mask*999999)
            # print('branching var value {}'.format(u_angle[w_branching_var]-l_angle[w_branching_var]))
        else:
            w_branching_var = None

        return z_branching_var, w_branching_var 


class BBenv(object):
    def __init__(self, observation_function=Observation, node_select_policy_path='default', policy_type='gnn', epsilon=0.001, init_U=None):
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
        
        self.global_L = 0 # global lower bound
        self.global_U = np.inf  # global upper bound        

        self.action_set_indices = None
        # current active node
        self.active_node = None

        self.global_U_ind = None
        self.failed_reward = -2000

        self.node_select_model = None

        self.init_U = 999999
        self.node_select_policy = self.default_node_select        
        
        self.z_incumbent = None
        self.w_incumbent = None
        
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
        self.global_L = 0 # global lower bound
        self.global_U = np.inf  # global upper bound        
        self.action_set_indices = None 
        self.active_node = None
        self.global_U_ind = None
        self.num_nodes = 1

        self.H = instance
        self.H_complex = self.H[0,:,:] + self.H[1,:,:]*1j
        
        self.min_bound_gap = np.ones(self.H.shape[-1])*0.01 # smallest size of continuous set to be branched on
        
        self.max_ant = max_ant

        # number of transmitters and users
        _, self.N, self.M = self.H.shape 
        self._is_reset = True
        self.action_set_indices_z = np.arange(1,self.N)
        self.action_set_indices_w = np.arange(1,self.M)

        # boolean vars corresponding to each antenna denoting its selection if True
        z_mask = np.zeros(self.N)
        # values of z (selection var) at the z_mask locations
        # for the root node it does not matter
        z_sol = np.zeros(self.N)

        done = False

        l = np.zeros(self.M)
        u = np.ones(self.M)*2*np.pi

        # initialize the root node 
        # try:
        [z, w, lower_bound, optimal] = qp_relaxed(self.H,
                                            l=l, 
                                            u=u, 
                                            z_mask=z_mask, 
                                            z_sol=z_sol, 
                                            max_ant=self.max_ant)
        self.global_L = lower_bound
        
        # Upper bound method
        self.z_incumbent = self.get_feasible_z(z)
        w_selected, obj, optimal = qp_relaxed_with_selected_antennas(self.H, l=l, u=u, z_sol=self.z_incumbent)
        w_feas = self.get_feasible_w(w_selected, self.z_incumbent)
        self.global_U = np.linalg.norm(w_feas*self.z_incumbent, 2)**2

        if not self.global_U == np.inf:
            self.w_incumbent = w_feas.copy()
        else:
            self.w_incumbent = np.zeros(self.H_complex.shape[0])

        self.active_node = Node(z_mask=z_mask, 
                                z_sol=z, 
                                z_feas=self.z_incumbent, 
                                w_sol = w, 
                                w_feas = w_feas,
                                l_angle=l,
                                u_angle=u,
                                U=self.global_U, 
                                L=lower_bound, 
                                depth=1, 
                                node_index=self.node_index_count)

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

        #TODO: re-write this to include both z and w oracle solutions
        self.optimal_angle = None
        if oracle_opt is not None:
            (self.oracle_z, oracle_w) = oracle_opt
            self.optimal_angle = np.angle(np.matmul(self.H_complex.conj().T, oracle_w))
            self.optimal_angle[self.optimal_angle<0] += 2*np.pi

        else:
            self.oracle_z = np.zeros(self.N)
            self.optimal_angle = np.random.randn(self.M, 1)

        self.H_energy_per_antenna = np.linalg.norm(self.H_complex, 2, axis=1)
        self.H_sum_per_antenna = np.abs(np.sum(self.H_complex, axis=1))

        return 

    def push_children(self, action_id, node_id):
        """
        action_id branching variable contains two indices, one for z and another for w
        use action_id branching variables to split the node into (possibly) four children
        if the action_id branching variables contain only one index and the other is None, then branch only on that variable
        e.g., (2,5) refers to second element of z and 5th element fo the continuous variable, c(5) = |h_5^H w|

        the sequence of children nodes are: left, midleft, midright, right
        """
        
        self.delete_node(node_id)

        z_action_id, c_action_id = action_id

        if z_action_id is None and c_action_id is None:
            # print('Both actions None')
            return
        # assert z_action_id is not None or c_action_id is not None, "Both branching variables set to None. Nothing to branch on"

        if z_action_id is not None:
            max_possible_ant = sum(self.active_node.z_mask*self.active_node.z_sol) + sum(1-self.active_node.z_mask)
            if max_possible_ant < self.max_ant:
                # print('**************got here if')

                # print('less than max ant antennas')
                return 
            elif max_possible_ant == self.max_ant:
                # print('**************got here else if')
                self.active_node.z_sol = self.active_node.z_mask*self.active_node.z_sol + (1-self.active_node.z_mask)*np.ones(self.N)
                self.active_node.z_mask = np.ones(self.N)
                z_action_id = None
            else:
                # print(self.active_node.z_mask, self.active_node.z_sol)
                z_mask_left = self.active_node.z_mask.copy()
                z_mask_left[z_action_id] = 1

                z_mask_right = self.active_node.z_mask.copy()
                z_mask_right[z_action_id] = 1

                z_sol_left = self.active_node.z_sol.copy()
                z_sol_left[z_action_id] = 0

                z_sol_right = self.active_node.z_sol.copy()
                z_sol_right[z_action_id] = 1

                if sum(z_sol_right*z_mask_right) == self.max_ant:
                    z_sol_right = z_sol_right*z_mask_right
                    z_mask_right = np.ones(self.N)

                # print('creating children z mask: {}, z_mask_left: {}, z_mask_right: {}, z_sol_left: {}, z_sol_right: {}'.format(self.active_node.z_mask, z_mask_left, z_mask_right, z_sol_left, z_sol_right))

        if c_action_id is not None:
            # if np.all(abs(self.active_node.u_angle - self.active_node.l_angle)> self.min_bound_gap):
            if self.active_node.u_angle[c_action_id] - self.active_node.l_angle[c_action_id] > min_bound_gap:
                mid_u_angle = self.active_node.u_angle.copy()
                mid_u_angle[c_action_id] = (mid_u_angle[c_action_id] + self.active_node.l_angle[c_action_id])/2
                mid_l_angle = self.active_node.l_angle.copy()
                mid_l_angle[c_action_id] = mid_u_angle[c_action_id]
            else:
                c_action_id = None
                if z_action_id is None:
                    print('No children at this node')
                    return
        
        children_sets = []
        if c_action_id is not None and z_action_id is not None:
            children_sets.append(((z_mask_left.copy() , z_sol_left.copy()), (self.active_node.l_angle.copy(), mid_u_angle.copy())))
            children_sets.append(((z_mask_left.copy() , z_sol_left.copy()), (mid_l_angle.copy(), self.active_node.u_angle.copy())))
            children_sets.append(((z_mask_right.copy(), z_sol_right.copy()), (self.active_node.l_angle.copy(), mid_u_angle.copy())))
            children_sets.append(((z_mask_right.copy(), z_sol_right.copy()), (mid_l_angle.copy(), self.active_node.u_angle.copy())))
            # for i in range(len(children_sets)):
            #     print('creating children {}'.format(children_sets[i][0][0]))

        elif c_action_id is not None and z_action_id is None:
            children_sets.append(((self.active_node.z_mask.copy(), self.active_node.z_sol.copy()), (self.active_node.l_angle.copy(), mid_u_angle.copy())))
            children_sets.append(((self.active_node.z_mask.copy(), self.active_node.z_sol.copy()), (mid_l_angle.copy(), self.active_node.u_angle.copy())))
        
        elif c_action_id is None and z_action_id is not None:
            children_sets.append(((z_mask_left.copy(), z_sol_left.copy()), (self.active_node.l_angle.copy(), self.active_node.u_angle.copy())))
            children_sets.append(((z_mask_right.copy(), z_sol_right.copy()), (self.active_node.l_angle.copy(), self.active_node.u_angle.copy())))
            
        children_stats = []

        if DEBUG:
            print('expanding node id {}, children {}'.format(self.active_node.node_index, (self.active_node.z_mask, self.active_node.z_sol, self.active_node.l_angle, self.active_node.u_angle)))
        for subset in children_sets:
            if DEBUG:
                print('\n creating children {}'.format(subset))
            children_stats.append(self.create_children(subset))
        
        if len(self.nodes) == 0:
            return

        # Update the global upper and lower bound 
        # update the incumbent solutions
        min_L_child = min([children_stats[i][1] for i in range(len(children_stats))])
        self.global_L = min(min(self.L_list), min_L_child)

        min_U_index = np.argmin([children_stats[i][0] for i in range(len(children_stats))])
        if self.global_U > children_stats[min_U_index][0]:
            # print('node depth at global U update {}'.format(self.active_node.depth + 1))
            self.global_U = children_stats[min_U_index][0] 
            self.z_incumbent = children_stats[min_U_index][2]
            self.w_incumbent = children_stats[min_U_index][3]
            

    def create_children(self, constraint_set):
        """
        Create the Node with the constraint set
        Compute the local lower and upper bounds 
        return the computed bounds for the calling function to update
        """
        (z_mask, z_sol), (l_angle, u_angle) = constraint_set 
        # check if the maximum number of antennas are already selected or all antennas are already assigned (z is fully assigned)
        if np.sum(z_mask*np.round(z_sol))==self.max_ant:
            z_sol = np.round(z_sol)*z_mask
            [w, L, optimal] = qp_relaxed_with_selected_antennas(self.H,
                                                                    l=l_angle,
                                                                    u=u_angle,
                                                                    z_sol=z_sol)
            # check this constraint                                                                    
            if not optimal:
                print('antennas: {} not optimal, may be infeasible'.format(None))
                # print('constraint self', constraint_set)   
                # print('parent', self.active_node.z_mask, self.active_node.z_sol, self.active_node.l_angle, self.active_node.u_angle)     

                                     
                return np.inf, np.inf, np.zeros(self.N), np.zeros(self.N)


            if L < self.active_node.L - self.epsilon:
                print('constraint self', constraint_set)   
                print('parent', self.active_node.z_mask, self.active_node.z_sol, self.active_node.l_angle, self.active_node.u_angle)         
                print(self.H)
                time.sleep(1)

            if not L >= self.active_node.L - self.epsilon:
                print('asserting', L >= self.active_node.L - self.epsilon, constraint_set, self.active_node.z_mask, self.active_node.z_sol, self.active_node.l_angle, self.active_node.u_angle, self.H)
                time.sleep(5)
            assert L >= self.active_node.L - self.epsilon, 'selected antennas: lower bound of child node less than that of parent'

            z_feas = z_sol.copy()
            w_feas = self.get_feasible_w(w,z_feas)
            U = self.get_objective(w_feas, z_feas)
            # create and append node
            self.node_index_count += 1
            new_node = Node(z_mask=z_mask,
                            z_sol=z_sol,
                            z_feas=z_feas,
                            w_sol=w,
                            w_feas=w_feas,
                            l_angle=l_angle,
                            u_angle=u_angle,
                            U=U,
                            L=L,
                            depth=self.active_node.depth+1,
                            node_index=self.node_index_count
                            )
            self.L_list.append(L)
            self.U_list.append(U)
            self.nodes.append(new_node)
            self.all_nodes.append(new_node)
            return U, L, z_feas, w_feas
        elif np.sum(z_mask*np.round(z_sol))>self.max_ant:
            return np.inf, np.inf, np.zeros(self.N), np.zeros(self.N)
        else:
            # print('solving relaxed with z_mask {}, z_sol {}'.format(z_mask, z_sol))
            # print('now solving relaxed problem')
            [z,w,L, optimal] = qp_relaxed(self.H,
                                    l=l_angle,
                                    u=u_angle,
                                    z_sol=z_sol,
                                    z_mask=z_mask,
                                    max_ant=self.max_ant,
                                    T=min(np.sqrt(self.global_U), 1000))

            
            # check this constraint                                                                    
            if not optimal:
                if DEBUG:
                    # print('relaxed: {} not optimal, may be infeasible'.format((self.H, z_mask, z_sol, l_angle, u_angle, self.max_ant, min(np.sqrt(self.global_U), 1000))))
                    print(z,w,L,optimal)
                else:
                    print('relaxed: not optimal, may be infeasible')

                # print('constraint self', constraint_set, L)   
                # print('parent', self.active_node.z_mask, self.active_node.z_sol, self.active_node.l_angle, self.active_node.u_angle, self.active_node.L)                                  
                return np.inf, np.inf, np.zeros(self.N), np.zeros(self.N)

            if L < self.active_node.L - self.epsilon:
                print('child node', constraint_set, L)   
                print('parent node', self.active_node.z_mask, self.active_node.z_sol, self.active_node.l_angle, self.active_node.u_angle, self.active_node.L)                                             
                print(self.H)

            
            assert L >= self.active_node.L - self.epsilon, 'relaxed: lower bound of child node less than that of parent'

            if not L == np.inf:
                # if the z is nearly determined round it
                temp = (1-z_mask)*(np.abs(z - 0.5))
                # print('z mask before', z_mask, z)
                z_mask[temp>0.499] = 1


                z = np.round(z_mask*z) + (1-z_mask)*z
                # print('z mask after', z_mask, z)

                z_feas = self.get_feasible_z(z)
                [w_feas_relaxed, L_feas_relaxed, optimal] =  qp_relaxed_with_selected_antennas(self.H,
                                                                    l=l_angle,
                                                                    u=u_angle,
                                                                    z_sol=z_feas)
                if optimal:
                    w_feas = self.get_feasible_w(w_feas_relaxed,z_feas)
                    U = self.get_objective(w_feas, z_feas)
                else:
                    w_feas = np.zeros(self.N)
                    U = np.inf
                # create and append node
                self.node_index_count += 1
                new_node = Node(z_mask=z_mask,
                                z_sol=z,
                                z_feas=z_feas,
                                w_sol=w,
                                w_feas=w_feas,
                                l_angle=l_angle,
                                u_angle=u_angle,
                                U=U,
                                L=L,
                                depth=self.active_node.depth+1,
                                node_index=self.node_index_count
                                )
                self.L_list.append(L)
                self.U_list.append(U)
                self.nodes.append(new_node)
                self.all_nodes.append(new_node)
                                                                                    
                return U, L, z_feas, w_feas
            
            else:
                return np.inf, np.inf, np.zeros(self.N), np.zeros(self.N)


    def get_feasible_w(self, w_selected, z_feas):
        # masked_w = ma.masked_array(abs(np.matmul(self.H_complex.conj().T, w_selected)), mask=z_feas)
        return w_selected/min(abs(np.matmul(self.H_complex.conj().T, w_selected*z_feas)))


    def get_feasible_z(self, z):
        # z_round = np.round(z)
        # if np.sum(z_round) <= self.max_ant:
        #     return z_round
        # else:
        mask = np.zeros(len(z))
        mask[np.argsort(z)[len(z)-self.max_ant:]] = 1
        return mask

    def get_objective(self, w, z_feas):
        return np.linalg.norm(w*z_feas, 2)**2

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

            self.active_node = self.nodes[node_id]
            break
        return node_id, self.observation_function().extract(self), self.is_optimal(self.active_node)


    def prune(self, observation):
        if isinstance(observation, Observation):
            observation = get_graph_from_obs(observation, self.action_set_indices_w)
        elif isinstance(observation, LinearObservation):
            observation = torch.tensor(observation.observation, dtype=torch.float32).unsqueeze(0)
        if self.node_select_policy == 'oracle':
            # print('prune called')
            return not self.is_optimal(self.active_node, debug=DEBUG)
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
                return True
            else:
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

    # This needs to be re-written for the current task
    def is_optimal(self, node, debug=False):
        if debug:
            print('id: {} oracle z: {}, oracle_angle: {}, node mask: {},  node z: {}, node angle: {}, {}'.format(node.node_index, self.oracle_z, self.optimal_angle, node.z_mask, node.z_sol, node.l_angle, node.u_angle))
            print('optimal node depth is {}'.format(node.depth))
        theta_feasibility = ((self.optimal_angle.squeeze()<=node.u_angle + 0.0001).all() and (self.optimal_angle.squeeze()>=node.l_angle-0.0001).all()) or ((self.optimal_angle.squeeze()+2*np.pi <= node.u_angle + 0.0001).all() and (self.optimal_angle.squeeze() +2*np.pi >= node.l_angl-0.0001).all())
        if np.linalg.norm(node.z_mask*(node.z_sol - self.oracle_z)) < 0.0001 and theta_feasibility :
            if debug:
                print('-- This node is optimal')
                print()
            return True
        else:
            if debug:
                print('-- This node is not optimal')
                print()
            return False


    # This needs to be re-written for the current task
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




def solve_bb(instance, max_ant=None, max_iter=10000, policy='default', policy_type='gnn', oracle_opt=None):
    t1 = time.time()
    if policy_type == 'default':
        env = BBenv(observation_function=Observation, epsilon=0.001)
    elif policy_type == 'gnn':
        env = BBenv(observation_function=Observation, epsilon=0.001)
    elif policy_type == 'linear':
        env = BBenv(observation_function=LinearObservation, epsilon=0.001)
    elif policy_type == 'oracle':
        env = BBenv(observation_function=Observation, epsilon=0.001)
        pass

    branching_policy = DefaultBranchingPolicy()

    t1 = time.time()

    env.reset(instance, max_ant=max_ant)
    timestep = 0
    done = False
    ub_list = []
    lb_list = []
    while timestep < max_iter and len(env.nodes)>0 and not done:
        print('timestep {}, U {}, L {}, len_nodes {}, depth_tree {}'.format(timestep, env.global_U, env.global_L, len(env.nodes), env.active_node.depth))
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
        branching_var = branching_policy.select_variable(node_feats, env.action_set_indices, env=env)

        # print(branching_var)
        # print('selected node z_sol {}, z_mask {}, z_feas {}'.format(env.nodes[node_id].z_sol, env.nodes[node_id].z_mask, env.nodes[node_id].z_feas))

        # last_id = len(env.nodes)
        done = env.push_children(branching_var, node_id)
        # for i in range(last_id, len(env.nodes)):
            # print('Children ', env.nodes[])

    
        timestep = timestep+1

        if env.is_terminal():
            break
        ub_list.append(env.global_U)
        lb_list.append(env.global_L)


    print('ended')
    print('result', env.z_incumbent.copy(), np.linalg.norm(env.w_incumbent*env.z_incumbent,2)**2)
    # returns the solution, objective value, timestep and the time taken
    return (env.z_incumbent.copy(), env.w_incumbent.copy()), env.global_U, timestep , time.time()-t1

if __name__ == '__main__':
    np.random.seed(seed = 100)
    N = 8
    M = 12
    max_ant = 3
    
    u_avg = 0
    t_avg = 0
    tstep_avg = 0
    for i in range(1):
        H = np.random.randn(N, M) + 1j*np.random.randn(N,M)    
        instance = np.stack((np.real(H), np.imag(H)), axis=0)
        _, global_U, timesteps, t = solve_bb(instance, max_ant=max_ant, max_iter = 10000)
        u_avg += global_U
        t_avg += t
        tstep_avg += timesteps

    print(u_avg, t_avg, tstep_avg, u_avg)

