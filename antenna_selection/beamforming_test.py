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
    
class Beamforming():
    def __init__(self, H, max_ant=None, T=1000, noise_var=1, min_snr=1):
        self.N, self.K = H.shape
        
        self.H = H.copy()
        self.max_ant = max_ant
        self.T = T
        self.zero = np.zeros(self.N)
        self.one = np.ones(self.N)
        
        self.c_1 = (1/np.sqrt(min_snr*noise_var))
        self.c_2 = (1/noise_var)

        self.z_constr = cp.Parameter(self.N)
        self.z_mask = cp.Parameter(self.N)
        
        self.W = cp.Variable((self.N, self.K), complex=True, name='W')
        self.z = cp.Variable((self.N), complex=False, name='z')
        self.obj = cp.Minimize(cp.square(cp.norm(self.W, 'fro')))

        self.constraints = []
        for k in range(self.K):
            Imask = np.eye(self.K)
            Imask[k,k] = 0
            self.constraints += [self.c_1*cp.real(np.expand_dims(self.H[:,k], axis=0) @ self.W[:,k]) >= cp.norm(cp.hstack((self.c_2*(self.W @ Imask).H @ self.H[:,k], np.ones(1))), 2)]
        self.constraints += [self.z >= self.zero, self.z <= self.one]
        
        self.constraints += [cp.sum(self.z) == self.max_ant] 
#         self.constraints += [self.z == cp.multiply(self.z_mask,self.z_sol)]
        self.constraints += [cp.multiply(self.z, self.z_mask) == self.z_constr]

#         for n in range(N):
#             if self.z_mask[n]:
#                 self.constraints += [self.z[n] == self.z_constr[n]]

        for k in range(self.K):
            self.constraints += [cp.real(self.W[:,k]) <=  T*self.z]
            self.constraints += [cp.real(self.W[:,k]) >= -T*self.z]
            self.constraints += [cp.imag(self.W[:,k]) <=  T*self.z]
            self.constraints += [cp.imag(self.W[:,k]) >= -T*self.z]


        # self.constraints += [cp.norm(self.W, 2, axis=1) <= self.T*self.z]
        self.prob = cp.Problem(self.obj, self.constraints)


    def solve_beamforming(self, z_mask=None, z_sol=None, W_init=None, z_init=None):
        if W_init is not None:
            self.W.value = W_init.copy()
        if z_init is not None:
            self.z.value = z_init.copy()
        
        self.z_mask.value = z_mask.copy()
        self.z_constr.value = (z_mask*z_sol).copy()
        try:
            self.prob.solve(solver=cp.MOSEK, verbose=False)
        except:
            return None, None, np.inf
            
        if self.prob.status in ['infeasible', 'unbounded']:
            print('infeasible solution')
            return None, None, np.inf

        return self.z.value, self.W.value, np.linalg.norm(self.W.value, 'fro')**2


class BeamformingWithSelectedAntennas():
    def __init__(self, H, max_ant=None, T=1000, noise_var=1, min_snr=1):
        self.N, self.K = H.shape
        
        self.H = H.copy()
        self.max_ant = max_ant
        self.T = T
        self.zero = np.zeros(self.N)
        self.one = np.ones(self.N)
        
        self.c_1 = cp.Parameter()
        self.c_1.value = (1/np.sqrt(min_snr*noise_var))
        self.c_2 = (1/noise_var)

        self.z_constr = cp.Parameter(self.N)
        
        self.W = cp.Variable((self.N, self.K), complex=True, name='W')
        self.obj = cp.Minimize(cp.square(cp.norm(self.W, 'fro')))

        self.constraints = []
        for k in range(self.K):
            Imask = np.eye(self.K)
            Imask[k,k] = 0
            self.constraints += [self.c_1*cp.real(np.expand_dims(self.H[:,k], axis=0) @ cp.multiply(self.W[:,k], self.z_constr)) >= cp.norm(cp.hstack((self.c_2*((self.W @ Imask).H @ cp.diag(self.z_constr)) @ self.H[:,k], np.ones(1))), 2)]
        
        # self.constraints += [cp.norm(self.W, 'inf', axis=1) <= self.T*self.z_constr]
        # for n in range(self.N):
        #     for k in range(self.K):
        #         self.constraints += [cp.real(self.W[n,k])<= T*self.z_constr[n]]
        #         self.constraints += [cp.real(self.W[n,k])>=-T*self.z_constr[n]]
        #         self.constraints += [cp.imag(self.W[n,k])<= T*self.z_constr[n]]
        #         self.constraints += [cp.imag(self.W[n,k])>=-T*self.z_constr[n]]
        self.prob = cp.Problem(self.obj, self.constraints)


    def solve_beamforming(self, z=None, W_init=None):
        if W_init is not None:
            self.W.value = W_init.copy()
        
        self.z_constr.value = np.round(z.copy())
        
        try:
            self.prob.solve(solver=cp.MOSEK, verbose=False)
        except:
            return None, np.inf
        if self.prob.status in ['infeasible', 'unbounded']:
            # print('infeasible antenna solution')
            return None, np.inf

        return self.W.value.copy(), np.linalg.norm(self.W.value.copy(), 'fro')**2



if __name__=='__main__':
    # N, K = 8,3
    # max_ant = 5
    # np.random.seed(150)
    # H = np.random.randn(N, K) + 1j*np.random.randn(N, K)

    # # z_sol = np.random.binomial(size=N, n=1, p= 0.5)
    # # z_mask = np.random.binomial(size=N, n=1, p=0.2)

    # z_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    # z_sol = np.array([1, 0, 1, 1, 1, 0, 0, 1])
    # # print(z_mask)
    
    # print(z_sol)
    # t1 = time.time()
    # z, W, obj = solve_beamforming_relaxed(H, max_ant=5, z_sol=z_sol, z_mask=z_mask)
    # print("TIME for completion: ", time.time()- t1)
    # print(H)
    # # t1 = time.time()
    # # z, W, obj = solve_beamforming_relaxed(H, max_ant=5, z_sol=z_sol, z_mask=z_mask, z_init=z, W_init=W)
    # # print("TIME for completion 2: ", time.time()- t1)

    # # obj2 = solve_beamforming_with_selected_antennas(H, z_sol)

    # # print(z)
    # # print(W)
    # print(obj)
    # # print(obj2)
    # # if obj2== np.inf:
    # # print('problem infeasible')
    pass