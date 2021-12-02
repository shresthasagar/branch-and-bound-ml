import torch
import numpy as np
from qp_relaxation_multicast import solve_qp_relaxed
from gnn_policy import GNNPolicy
from gnn_dataset import get_graph_from_obs
from fpp_sca import fpp_sca

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
        self.antenna_features = np.concatenate((
                                    model.active_node.w_sol.real, 
                                    model.active_node.w_sol.imag, 
                                    abs(model.active_node.w_sol),
                                    model.active_node.w_feas.real, 
                                    model.active_node.w_feas.imag, 
                                    abs(model.active_node.w_feas),
                                    model.w_opt.real, 
                                    model.w_opt.imag, 
                                    abs(model.w_opt)), axis=1)
        
        # edge features
        self.edge_index = np.stack((np.repeat(np.arange(model.N), model.M), np.tile(np.arange(model.M), model.N)))
        self.edge_features = np.zeros((model.M*model.N, 3))
        self.edge_features[:,0] = np.real(model.H_complex.reshape(-1))
        self.edge_features[:,1] = np.imag(model.H_complex.reshape(-1))
        self.edge_features[:,2] = np.abs(model.H_complex.reshape(-1))

        # construct variable features
        # global features
        self.variable_features = np.zeros((model.M, 10))
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
        
        self.variable_features = np.zeros((model.M, 9))
        self.candidates = model.action_set_indices

        # construct the observation
        H_w = np.matmul(model.H_complex.conj().T, model.active_node.w_sol)
        self.variable_features[:,7] = np.squeeze(np.abs(H_w))
        
        angle_w_sol = np.angle(model.active_node.w_sol)
        angle_w_sol[angle_w_sol<0] += 2*np.pi

        angle_w_feas = np.angle(model.active_node.w_feas)
        angle_w_feas[angle_w_feas<0] += 2*np.pi
        
        angle_w_opt = np.angle(model.w_opt)
        angle_w_opt[angle_w_opt<0] += 2*np.pi

        self.observation = np.concatenate((model.active_node.w_sol.real.reshape(-1), 
                                            model.active_node.w_sol.imag.reshape(-1), 
                                            abs(model.active_node.w_sol).reshape(-1),
                                            angle_w_sol.reshape(-1),
                                            
                                            model.active_node.w_feas.real.reshape(-1), 
                                            model.active_node.w_feas.imag.reshape(-1), 
                                            abs(model.active_node.w_feas).reshape(-1),
                                            angle_w_feas.reshape(-1),
                                            
                                            model.w_opt.real.reshape(-1), 
                                            model.w_opt.imag.reshape(-1), 
                                            abs(model.w_opt).reshape(-1),
                                            angle_w_opt.reshape(-1)))

        self.observation = np.concatenate((self.observation, 
                                            np.real(model.H_complex.reshape(-1)), 
                                            np.imag(model.H_complex.reshape(-1)), 
                                            np.abs(model.H_complex.reshape(-1))))
        
        self.observation = np.concatenate((self.observation, 
                                            np.array([model.global_U]), 
                                            np.array([model.global_L]),
                                            np.array([model.active_node.depth]),
                                            np.array(model.active_node.lb),
                                            np.array(model.active_node.ub),
                                            np.squeeze(np.real(H_w)),
                                            np.squeeze(np.imag(H_w)),
                                            np.squeeze(np.abs(H_w)),
                                           (np.squeeze(np.abs(H_w))>=1)*1 ))
        # self.observation = np.squeeze(np.abs(H_w))
        self.candidates = model.action_set_indices

        #TODO: include the normalized number of times a variable has been selected by the current branching policy    
        return self
