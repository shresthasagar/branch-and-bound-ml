import cvxpy as cp
import numpy as np

def sdr(H, num_sample_random=50):
    N, M = H.shape
    
    W = cp.Variable((N,N), hermitian=True)
    constraints = [W >> 0]
    for i in range(M):
        HH = np.matmul(H[:,i:i+1], H[:,i:i+1].conj().T)
        constraints += [cp.real(cp.trace(HH @ W)) >= 1]
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(W))), constraints)
    prob.solve()

    # Randomization
    W_real = np.real(W.value)
    W_imag = np.imag(W.value)

    # randA
    lmbda, U = np.linalg.eig(W.value)
    lmbda = np.abs(np.real(lmbda))
    randvecs = [np.matmul(U, np.matmul(np.diag(np.sqrt(lmbda)), np.exp(1j*np.random.rand(8,1)*2*np.pi))) for _ in range(num_sample_random)]
    outvecs = [ vec/min(abs(np.matmul(H.conj().T, vec))) for vec in randvecs]
    # sol_id = np.argmin(norms)


    # randB
    randvecs = [ np.sqrt(np.real(np.expand_dims(np.diag(W.value),axis=1)))*np.exp(1j*np.random.rand(8,1)*2*np.pi) for _ in range(num_sample_random)]
    outvecs += [ vec/min(abs(np.matmul(H.conj().T, vec))) for vec in randvecs]

    # randC
    
    randvecs = [np.random.multivariate_normal(np.zeros(N), W_real) + 1j* np.random.multivariate_normal(np.zeros(N), W_imag) for i in range(num_sample_random)]
    outvecs += [ vec/min(abs(np.matmul(H.conj().T, vec))) for vec in randvecs]
    norms = [np.linalg.norm(vec) for vec in outvecs]
    sol_id = np.argmin(norms)

    return outvecs[sol_id]

def sla(H):
    N, M = H.shape
    
    z = np.random.randn(N,1) + 1j*np.random.randn(N,1)
    z = z/min(abs(np.matmul(H.conj().T, z)))
    p = np.matmul(H.conj().T, z)

    v = cp.Variable((M,1), complex=True)
    w = cp.Variable((N,1), complex=True)
    
    
    obj = cp.Minimize(cp.square(cp.norm(w)))
    maxIter = 10
    iter = 0
    power_chage = 100
    while  iter < 10 or   :
        constraints = []
        for i in range(M):
            constraints += [np.linalg.norm(p[i])**2 + 2*(cp.real(p[i])*(cp.real(v[i])-cp.real(p[i])) + cp.imag(p[i])*(cp.imag(v[i])-cp.imag(p[i]))) >= 1]
            constraints += [cp.real(v[i]) == cp.real(np.expand_dims(H[i,:], axis=1).conj().T @ w)]
            constraints += [cp.imag(v[i]) == cp.imag(np.expand_dims(H[i,:], axis=1).conj().T @ w)]

        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)
        p = v.value.copy()
    return w.value


if __name__=='__main__':
    from fpp_sca import fpp_sca
    from acr_bb import solve_bb

    N,M = 8,8
    
    sdr_ogap = []
    fpp_ogap = []
    sla_ogap = []

    for i in range(50):
        H = np.random.randn(N,M) + 1j*np.random.randn(N,M)
        w_fpp = sdr(H)
        w_sdr = sdr(H, 1000)
        w_sla = sla(H)

        instance = np.stack((np.real(H), np.imag(H)), axis=0)
        w_bb, _ = solve_bb(instance, max_iter=10000)
        bb_obj = np.linalg.norm(w_bb)**2

        sdr_obj = np.linalg.norm(w_sdr)**2
        fpp_obj = np.linalg.norm(w_fpp)**2
        sla_obj = np.linalg.norm(w_sla)**2 

        print(min(abs(np.matmul(H.conj().T, w_sdr))) , sdr_obj)
        print(min(abs(np.matmul(H.conj().T, w_sla))) , sdr_obj)
        print(min(abs(np.matmul(H.conj().T, w_fpp))), fpp_obj)

        print(min(abs(np.matmul(H.conj().T, w_bb))), bb_obj)

        sdr_ogap.append(abs((sdr_obj-bb_obj)/bb_obj)*100)
        fpp_ogap.append(abs((fpp_obj-bb_obj)/bb_obj)*100)
        sla_ogap.append(abs((sla_obj-bb_obj)/bb_obj)*100)


    print('sdr ogap is {}'.format(np.mean(sdr_ogap)))
    print('fpp ogap is {}'.format(np.mean(fpp_ogap)))
    print('sla ogap is {}'.format(np.mean(sla_ogap)))



