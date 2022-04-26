import cvxpy as cp
import numpy as np
from robust_beamforming.solve_relaxation import *

class ReweightedPenalty:
    def __init__(self, 
                H=None, 
                gamma=1, 
                sigma_sq=2, 
                epsi=0.3, 
                max_ant= None):

        self.H = H.copy()
        self.N, self.M = H.shape
        self.sigma_sq= sigma_sq*np.ones(self.M)
        self.gamma= gamma*np.ones(self.M) #SINR levels, from -10dB to 20dB
        self.epsi= epsi*np.ones(self.M)

        self.max_ant = max_ant

        self.epsilon = 1e-5
        self.lmbda_lb = 0
        self.lmbda_ub = 1e6
    
    def solve(self):
        # step 1
        U = np.ones((self.N, self.N))
        U_new = np.ones((self.N, self.N))
        
        r = 0
        max_iter = 30
        # while np.linalg.norm(u-u_new)>0.0001 and r < max_iter:
        while r < max_iter:
            print('sparse iteration  {}'.format(r))
            r += 1
            U = U_new.copy()
            X_tilde = self.sparse_iteration(U)

            if X_tilde is None:
                return None, np.zeros(self.N)

            a = np.diag(X_tilde)
            mask = (a>0.01)*1
            if mask.sum()<= self.max_ant:
                print('Sparse enough solution found')
                break
            U_new = 1/(X_tilde + self.epsilon)

        prelim_mask = mask.copy()
        before_iter_ant_count = mask.sum()
        print('mask after sparse iteration {}'.format(mask))
        if mask.sum() > self.max_ant:
            # sparse enough solution not found!
            return None, mask.copy()

        # step 2
        r = 0
        max_iter = 30
        while mask.sum() != self.max_ant and r < max_iter:
            r += 1
            lmbda = self.lmbda_lb + (self.lmbda_ub - self.lmbda_lb)/2
            X_tilde = self.solve_sdps_with_soft_as(lmbda, U_new)

            if X_tilde is None:
                return None, np.zeros(self.N)

            a = np.diag(X_tilde)
            mask = (a>0.01)*1
            print('iteration {}'.format(r), lmbda, mask.sum(), self.lmbda_lb, self.lmbda_ub)
            if mask.sum() == self.max_ant:
                break
            elif mask.sum() > self.max_ant:
                self.lmbda_lb = lmbda
            elif mask.sum() < self.max_ant:
                self.lmbda_ub = lmbda
        if mask.sum()>self.max_ant:
            mask = prelim_mask.copy()    
        print('num selected antennas', mask.sum())

        after_iter_ant_count = mask.sum()
        print(mask)
        # step 3
        _, W, obj, optimal = solve_rsdr(H=self.H, 
                                        z_mask=np.ones(self.N), 
                                        z_sol=mask )
        print(obj)
        print('Before lambda iteration: {}'.format(before_iter_ant_count))
        print('After lambda iteration: {}'.format(after_iter_ant_count))
        
        if mask.sum() > self.max_ant:
            return None, mask.copy()
        return obj.copy(), mask.copy()


    def sparse_iteration(self, U):
        
        X = []
        for i in range(self.M):
            X.append(cp.Variable((self.N, self.N), hermitian=True))
        X_tilde = cp.Variable((self.N, self.N), complex=False)
        t = cp.Variable(self.M)

        obj = cp.Minimize(cp.trace(U @ X_tilde))
        
        constraints = []
        for m in range(self.M):
            Q = (1+1/self.gamma[m])*X[m] - cp.sum(X)
            r = Q @ self.H[:,m:m+1]
            s = self.H[:,m:m+1].conj().T @ Q @ self.H[:,m:m+1] - self.sigma_sq[m:m+1]
            Z = cp.hstack((Q+t[m]*np.eye(self.N), r))
            # Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-t[m:m+1]*epsi[m:m+1]**2 ))))
            Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-cp.multiply(t[m:m+1], self.epsi[m:m+1]**2) ))))

            constraints += [X[m] >> 0]
            constraints += [Z >> 0]
            constraints += [X_tilde >= cp.abs(X[m])]
            constraints += [t >= 0]
        
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)

        if prob.status in ['infeasible', 'unbounded']:
            # print('infeasible antenna solution')
            # return np.ones((self.N, self.N))
            return None

        return X_tilde.value


    def solve_sdps_with_soft_as(self, lmbda, U):
        X = []
        for i in range(self.M):
            X.append(cp.Variable((self.N, self.N), hermitian=True))
        X_tilde = cp.Variable((self.N, self.N), complex=False)
        t = cp.Variable(self.M)

        obj = cp.Minimize(cp.real(cp.sum([cp.trace(Xi) for Xi in X])) + lmbda*cp.trace(U @ X_tilde))
        
        constraints = []
        for m in range(self.M):
            Q = (1+1/self.gamma[m])*X[m] - cp.sum(X)
            r = Q @ self.H[:,m:m+1]
            s = self.H[:,m:m+1].conj().T @ Q @ self.H[:,m:m+1] - self.sigma_sq[m:m+1]
            Z = cp.hstack((Q+t[m]*np.eye(self.N), r))
            # Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-t[m:m+1]*epsi[m:m+1]**2 ))))
            Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-cp.multiply(t[m:m+1], self.epsi[m:m+1]**2) ))))

            constraints += [X[m] >> 0]
            constraints += [Z >> 0]
            constraints += [X_tilde >= cp.abs(X_tilde)]
            constraints += [t >= 0]
        
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)

        if prob.status in ['infeasible', 'unbounded']:
            # print('infeasible antenna solution')
            # return np.ones((self.N, self.N))
            return None

        return X_tilde.value

if __name__=='__main__':
    N, M, L = 8,4,3

    H = (np.random.randn(N,M) + 1j*np.random.randn(N,M))/np.sqrt(2)

    # H =np.array([[ 0.1414 + 0.3571j,   0.3104 + 0.8294j,  -0.2649 + 0.7797j],
    #     [-0.7342 + 0.0286j,  -0.0290 - 0.1337j,  -1.1732 - 0.2304j],
    #     [ 0.3743 - 0.1801j,   0.2576 + 0.0612j,  -0.2194 - 0.4121j],
    #     [-1.5677 - 0.3873j,  -0.5400 + 1.0695j,  -0.4384 + 0.5315j],
    #     [ 0.0760 + 1.1855j,   0.3886 + 1.2680j,  -0.7245 - 0.6108j]])
    import time

    t1 = time.time()
    iterIns = ReweightedPenalty(H=H, max_ant=L)
    obj, mask = iterIns.solve()
    time_taken = time.time()-t1

    print(obj)
    print('time taken', time_taken)