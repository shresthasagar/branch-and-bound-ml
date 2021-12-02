import numpy as np
import itertools
from beamforming import solve_beamforming_with_selected_antennas

def enumeration_antenna_selection(H, max_ant):
    N, M = H.shape
    index_set  = np.arange(N)
    best_obj_val = 1000
    best_mask = np.zeros(N)
    for i in range(max_ant+1):
        print(i)
        for subset in itertools.combinations(index_set, i):
            mask = np.zeros(N)
            mask[list(subset)] = 1
            # mask[list((0,3,10))] = 1
            obj_val = solve_beamforming_with_selected_antennas(H, mask)
            print(subset, best_obj_val, obj_val)
            
            if best_obj_val > obj_val:
                best_obj_val = obj_val
                best_mask = mask.copy()
    return best_obj_val, best_mask


if __name__ == '__main__':
    N = 12
    M = 4
    max_ant = 5
    np.random.seed(seed = 100)
    H = np.random.randn(N,M) + 1j*np.random.randn(N,M)
    f, m = enumeration_antenna_selection(H, max_ant)
    print(f,m)


            