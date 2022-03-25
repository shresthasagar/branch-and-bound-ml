import numpy as np

class Observation(object):
    def __init__(self):
        self.antenna_features  = None # np.zeros(N, 3) # three features for each antenna
        self.variable_features = None # np.zeros(M, 15)
        self.edge_index = None
        self.edge_features     = None # np.zeros(N*M, 3)
        self.candidates        = None # np.arange(M)
        pass


    def check_integrality(self, z_sol, z_mask):
        sum_z= np.sum(np.abs(z_mask*(z_sol - np.round(z_sol))))
        return sum_z < 0.001

    def extract(self, model):
        # TODO: make the observation out of the model 
        # if not self.check_integrality(model.active_node.z_sol, model.active_node.z_mask):
        #     print('z sol and mask in obs', model.active_node.z_sol, model.active_node.z_mask)
        assert self.check_integrality(model.active_node.z_sol, model.active_node.z_mask)

        self.candidates = model.action_set_indices
        self.antenna_features = np.zeros((model.N,13))
        self.antenna_features[:,0] = model.active_node.z_sol
        self.antenna_features[:,1] = model.active_node.z_feas
        self.antenna_features[:,2] = model.active_node.z_mask

        self.antenna_features[:,4] = model.active_node.w_sol.real
        self.antenna_features[:,5] = model.active_node.w_sol.imag
        self.antenna_features[:,6] = abs(model.active_node.w_sol)

        self.antenna_features[:,7] = model.active_node.w_feas.real
        self.antenna_features[:,8] = model.active_node.w_feas.imag
        self.antenna_features[:,9] = abs(model.active_node.w_feas)

        self.antenna_features[:,10] = model.w_incumbent.real
        self.antenna_features[:,11] = model.w_incumbent.imag
        self.antenna_features[:,12] = abs(model.w_incumbent)
        
        # edge features
        self.edge_index = np.stack((np.repeat(np.arange(model.N), model.M), np.tile(np.arange(model.M), model.N)))
        self.edge_features = np.zeros((model.M*model.N, 3))
        self.edge_features[:,0] = np.real(model.H_complex.reshape(-1))
        self.edge_features[:,1] = np.imag(model.H_complex.reshape(-1))
        self.edge_features[:,2] = np.abs(model.H_complex.reshape(-1))

        # construct variable features
        # global features
        global_upper_bound = -1 if model.global_U == np.inf else model.global_U
        self.variable_features = np.zeros((model.M, 10))
        self.variable_features[:,0] = model.global_L # global lower bound
        self.variable_features[:,1] = global_upper_bound # global upper bound
        self.variable_features[:,2] = (model.active_node.U - global_upper_bound) < model.epsilon 
        self.variable_features[:,3] = model.active_node.l_angle # constraint lower bound of the selected node
        self.variable_features[:,4] = model.active_node.u_angle # constraint upper bound of the selected node

        # local features
        w_determined_from_relaxed = (model.active_node.z_mask*model.active_node.z_sol)*model.active_node.w_sol + (1 - model.active_node.z_mask)*model.active_node.w_sol
        
        H_w = np.matmul(model.H_complex.conj().T, w_determined_from_relaxed)
        
        self.variable_features[:,5] = np.squeeze(np.real(H_w))
        self.variable_features[:,6] = np.squeeze(np.imag(H_w))
        self.variable_features[:,7] = np.squeeze(np.abs(H_w))
        self.variable_features[:,8] = (np.squeeze(np.abs(H_w))>=1)*1
        self.variable_features[:,9] = model.active_node.depth

        #TODO: include the normalized number of times a variable has been selected by the current branching policy    
        return self


class LinearObservation(object):
    """
    Constructs a long obervation vector for linear neural network mapping
    """

    def __init__(self):
        self.observation = None 
        self.candidates  = None # np.arange(M)
        self.variable_features = None
        pass

    def extract(self, model):
        return self