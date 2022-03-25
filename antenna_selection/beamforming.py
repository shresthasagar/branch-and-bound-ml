import cvxpy as cp
import numpy as np
import time

def beamforming(H, z, noise_var=1, min_snr=1):
    N, K = H.shape
    W = cp.Variable((N,K), complex=True)
    obj = cp.Minimize(cp.square(cp.norm(W, 'fro')))

    c_1 = (1/np.sqrt(min_snr*noise_var))
    c_2 = (1/noise_var)

    mask = np.diag(z.squeeze().copy())
    constraints = []
    for k in range(K):
        constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0) @ W[:,k]) >= cp.norm(cp.hstack((c_2*(W.H @ mask) @ H[:,k], np.ones(1))), 2)]
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True)
    return W.value

def solve_beamforming_relaxed(H, max_ant=None, z_mask=None, z_sol=None, M=1000, noise_var=1, min_snr=1):    
    """
    Solves the relaxed formulation of the boolean problem
    """
    # print('z mask: {},\n z value: {}'.format(z_mask, z_sol))
    t1 = time.time()

    N, K = H.shape
    W = cp.Variable((N,K), complex=True)
    z = cp.Variable((N), complex=False)
    obj = cp.Minimize(cp.square(cp.norm(W, 'fro')))

    zero = np.zeros(N)
    one = np.ones(N)
    c_1 = (1/np.sqrt(min_snr*noise_var))
    c_2 = (1/noise_var)

    constraints = []
    for k in range(K):
        Imask = np.eye(K)
        Imask[k,k] = 0
        constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0) @ W[:,k]) >= cp.norm(cp.hstack((c_2*(W @ Imask).H @ H[:,k], np.ones(1))), 2)]

    # for k in range(K):
    #     constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0) @ W[:,k]) >= cp.norm(cp.hstack((c_2*W.H @ H[:,k], np.ones(1))), 2)]
    constraints += [z >= zero, z <= one]
    constraints += [cp.sum(z) <= max_ant] 
    # constraints += [np.diag(z_mask) @ z == z_sol]
    # for n in range(N):
    #     if z_mask[n]:
    #         constraints += [z[n] == np.round(z_sol[n])]
    constraints += [cp.multiply(z, z_mask) == (z_sol*z_mask).copy()]

    #TODO: write the below in vectorized form
    # for n in range(N):
    #     # constraints += [cp.norm(W[n,:], 2)<= M*z[n]]
    #     for k in range(K):
    #         constraints += [cp.real(W[n,k])<= M*z[n]]
    #         constraints += [cp.real(W[n,k])>=-M*z[n]]
    #         constraints += [cp.imag(W[n,k])<= M*z[n]]
    #         constraints += [cp.imag(W[n,k])>=-M*z[n]]
    # constraints += [cp.norm(W, 1, axis=1) <= M*z]
    
    # zu = z.expand_dims(axis=1)
    for k in range(K):
        constraints += [cp.real(W[:,k]) <= M*z]
        constraints += [cp.real(W[:,k]) >= -M*z]
        constraints += [cp.imag(W[:,k]) <= M*z]
        constraints += [cp.imag(W[:,k]) >= -M*z]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    if prob.status in ['infeasible', 'unbounded']:
        print('infeasible solution')
        return None, None, np.inf

    return z.value, W.value, np.linalg.norm(W.value, 'fro')**2

def solve_beamforming_with_selected_antennas(H, z, M=1000, noise_var=1, min_snr=1):    
    """
    Solves the beamforming problem given boolean z (antenna selection variables)
    """
    # print('feasible z: {}'.format(z))
    N, K = H.shape
    W = cp.Variable((N,K), complex=True)
    obj = cp.Minimize(cp.square(cp.norm(W, 'fro')))

    c_1 = (1/np.sqrt(min_snr*noise_var))
    c_2 = (1/noise_var)

    z_matrix = np.diag(z)
    constraints = []
    for k in range(K):
        Imask = np.eye(K)
        Imask[k,k] = 0
        constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0) @ cp.multiply(W[:,k], z)) >= cp.norm(cp.hstack((c_2*((W @ Imask).H @ z_matrix) @ H[:,k], np.ones(1))), 2)]

        # constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0) @ W[:,k]) >= cp.norm(cp.hstack((c_2*(W @ Imask).H @ H[:,k], np.ones(1))), 2)]

    # for k in range(K):
    #     constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0) @ W[:,k]) >= cp.norm(cp.hstack((c_2*W.H @ H[:,k], np.ones(1))), 2)]

    #TODO: write the below in vectorized form
    
    # for n in range(N):
    #     for k in range(K):
    #         constraints += [cp.real(W[n,k])<=M*z[n]]
    #         constraints += [cp.real(W[n,k])>=-M*z[n]]
    #         constraints += [cp.imag(W[n,k])<=M*z[n]]
    #         constraints += [cp.imag(W[n,k])>=-M*z[n]]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    # print("z", z)

    if prob.status in ['infeasible', 'unbounded']:
        # print('infeasible solution with selected antennas')
        return None, np.inf
    # print(z,np.linalg.norm(W.value, 'fro')**2)
    return W.value, np.linalg.norm(W.value, 'fro')**2


if __name__=='__main__':
    N, K = 8,8
    max_ant = 5
    np.random.seed(150)
    H = np.random.randn(N, K) + 1j*np.random.randn(N, K)
    z_sol = np.random.binomial(size=N, n=1, p= 0.5)
    z_mask = np.random.binomial(size=N, n=1, p=0.2)

    z_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    z_sol = np.array([1, 0, 1, 1, 1, 0, 0, 1])
    # print(z_mask)
    print(z_sol)
    t1 = time.time()
    z, W, obj = solve_beamforming_relaxed(H, max_ant=5, z_sol=z_sol, z_mask=z_mask)
    print("TIME for completion: ", time.time()- t1)
    # obj2 = solve_beamforming_with_selected_antennas(H, z_sol)

    # print(z)
    # print(W)
    print(obj)
    # print(obj2)
    # if obj2== np.inf:
    #     print('problem infeasible')
    pass