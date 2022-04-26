"""
1. Merge the two observation into 1
2. Design a GNN to work on the merged observation object
3. Write the training code 

"""
import numpy as np
from robust_beamforming.observation import Observation, LinearObservation


class MergedObservation:
    def __init__(self, observation1, observation2):
        N = observation1.antenna_features.shape[0]
        M = observation1.variable_features.shape[0]

        # antenna features
        self.antenna_features = np.zeros(N, 3)
        
        # Edge Features

        # Variable Features 
        