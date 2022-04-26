import numpy as np

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
        self.antenna_features = np.zeros((model.N,4))
        self.antenna_features[:,0] = model.active_node.z_sol
        self.antenna_features[:,1] = model.active_node.z_feas
        self.antenna_features[:,2] = model.active_node.z_mask
        self.antenna_features[:,3] = np.linalg.norm(model.active_node.W_sol,  axis=1)

        # edge features
        self.edge_index = np.stack((np.repeat(np.arange(model.N), model.M), np.tile(np.arange(model.M), model.N)))
        self.edge_features = np.zeros((model.M*model.N, 9))
        self.edge_features[:,0] = np.real(model.H_complex.reshape(-1))
        self.edge_features[:,1] = np.imag(model.H_complex.reshape(-1))
        self.edge_features[:,2] = np.abs(model.H_complex.reshape(-1))

        self.edge_features[:,3] = np.real(model.W_incumbent.reshape(-1))
        self.edge_features[:,4] = np.imag(model.W_incumbent.reshape(-1))
        self.edge_features[:,5] = np.abs(model.W_incumbent.reshape(-1))

        self.edge_features[:,6] = np.real(model.active_node.W_sol.reshape(-1))
        self.edge_features[:,7] = np.imag(model.active_node.W_sol.reshape(-1))
        self.edge_features[:,8] = np.abs(model.active_node.W_sol.reshape(-1))


        # construct variable features
        # global features
        global_upper_bound = 1000 if model.global_U == np.inf else model.global_U
        local_upper_bound =  2000 if model.active_node.U == np.inf else model.active_node.U

        self.variable_features = np.zeros((model.M, 8))
        self.variable_features[:,0] = model.global_L # global lower bound
        self.variable_features[:,1] = global_upper_bound # global upper bound
        self.variable_features[:,2] = (model.active_node.U - global_upper_bound) < model.epsilon 


        # local features
        W_H = np.matmul(model.active_node.W_sol.conj().T, model.H_complex)
        W_H = np.abs(W_H)
        mask = np.eye(*W_H.shape)
        mask_comp = 1-mask
        direct = np.sum(W_H*mask, axis=1)
        interference = W_H*mask_comp
        aggregate_interference = np.sum(interference, axis=1)

        H_w = np.matmul(model.H_complex.conj().T, model.active_node.W_sol)
        self.variable_features[:,3] = np.squeeze(direct)
        self.variable_features[:,4] = np.squeeze(aggregate_interference)
        self.variable_features[:,5] = model.active_node.depth

        # TODO: include local lower and upper bounds
        # TODO: include other bb statistics and decision variables
        self.variable_features[:,6] = local_upper_bound
        self.variable_features[:,7] = model.active_node.L
        # self.variable_features[:,8] = 

        

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