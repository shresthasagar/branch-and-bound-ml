import cvxpy as cp
import numpy as np

def fpp_sca(H):
    N, M = H.shape
    z = np.random.randn(N,1) + 1j*np.random.randn(N,1)
    z = z/min(abs(np.matmul(H.conj().T, z)))

    w = cp.Variable((N,1), complex=True)
    s = cp.Variable((M,1), complex=False)
    
    obj = cp.Minimize(cp.square(cp.norm(w)) + 1*cp.norm(s,1) )
    for _ in range(30):
        constraints = []
        for i in range(M):
            HH = np.expand_dims(H[:,i], axis=1)
            HH_nsd = - np.matmul(HH, HH.conj().T)
            constraints += [2*cp.real(np.matmul(z.conj().T, HH_nsd) @ w) <= -1 + cp.real( cp.quad_form(z, HH_nsd) ) + s[i]]
            constraints += [s[i] >=0 ]

        prob = cp.Problem(obj, constraints)
        prob.solve()
        z = w.value.copy()
    return w.value


if __name__=='__main__':
    N,M = 16,16
    H = np.random.randn(N,M) + 1j*np.random.randn(N,M)
    w = fpp_sca(H)
    print(np.linalg.norm(w)**2)
    print('here')
    print(w)
    print(abs(np.matmul(H.conj().T, w)))