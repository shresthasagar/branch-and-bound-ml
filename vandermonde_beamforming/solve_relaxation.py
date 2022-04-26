import numpy as np
import cvxpy as cp

class VandermondeChannel:
    def __init__(self, 
                random = True,
                phi = None, 
                N=6, 
                M=12, 
                L=None,
                SINRdB=10,
                G=2,
                n_beams_per_group=2):
        assert M % G == 0, 'cannot divide into groups with equal number of users'
        assert G >= 1, 'Number of groups cannot be less than 1' 
        assert M % (G*n_beams_per_group) == 0, 'Number of users in a group cannot be divided equally into different beams'

        SINR = 10**(SINRdB/10)*np.ones(M)
        sigma2 = np.ones(M)

        if phi is not None: 
            assert len(phi) == M, 'number of user locations not equal to the number of users'
        if not random: 
            if phi is not None:
                theta = -np.pi*np.sin(phi*np.pi/180) 
            else:
                phi = np.random.rand(M)*2*np.pi
                phi = np.sort(phi)
                perm = np.random.permutation(np.arange(n_beams_per_group*G))
                phi_new = np.array([])
                nbeams = n_beams_per_group*G
                for i in perm:
                    phi_new = np.concatenate((phi_new, phi[int(i*M/nbeams):int((i+1)*M/nbeams)]))
                phi = phi_new.copy()
                theta = -np.pi*np.sin(phi)
        else:
            phi = np.random.rand(M)*2*np.pi
            theta = -np.pi*np.sin(phi)
        
        Gk = M/G*np.ones((G,1))
        csGk = np.cumsum(Gk)

        group_membership = np.zeros(Gk[0])
        for k in range(1,csGk):
            group_membership = np.concatenate(group_membership, np.ones(csGk[k])*k)

        H = self.v_theta(theta, N)

        A = np.zeros(M, G*N)
        for i in range(M):
            a = -SINR[i]*sigma2[i]*np.ones(G)
            a[group_membership(i)] = 1
            A[i, :] = np.kron(a, H[:,i])

        c = np.zeros(G*N + 2*M + G*N*N, 1)
        c[1:N:G*N] = 1

        b = np.concatenate(SINR*sigma2, np.zeros(G*N))
        A = np.concatenate((A, 1j*np.eye(M), -np.eye(M), np.zeros(M, G*N*N)), axis=1)
        
        allE = np.zeros(N, N*N)
        for n in range(N):
            E = np.diag(np.ones(N-n),-n)
            allE[n,:] = -E.flatten()

        subA = np.concatenate((np.eye(G*N), np.zeros(G*N, 2*M), np.kron(np.eye(G), allE)), axis=1)
        A = np.concatenate((A, subA), axis=0)
        
        return H
    
    def solve_beamforming(H, G, group_membership, SINR, sigma2):
        N,M = H.shape
        
        r = cp.variable((G*N), complex=True)
        x = cp.variable((G*N*N), complex=True)

        A = np.zeros(M, G*N)
        for i in range(M):
            a = -SINR[i]*sigma2[i]*np.ones(G)
            a[group_membership(i)] = 1
            A[i, :] = np.kron(a, H[:,i])

        c = np.zeros(G*N + 2*M + G*N*N, 1)
        c[1:N:G*N] = 1

        b = np.concatenate(SINR*sigma2, np.zeros(G*N))
        A = np.concatenate((A, 1j*np.eye(M), -np.eye(M), np.zeros(M, G*N*N)), axis=1)
        
        allE = np.zeros(N, N*N)
        for n in range(N):
            E = np.diag(np.ones(N-n),-n)
            # allE[n,:] = -E.flatten()
            allE[n,:] = -cp.vec(E)
            

        subA = np.concatenate((np.eye(G*N), np.zeros(G*N, 2*M), np.kron(np.eye(G), allE)), axis=1)
        A = np.concatenate((A, subA), axis=0)

        obj = cp.minimize(cp.real(cp.sum(cp.multiply(c[:G*N], r))))

        constraints = []

        constraints.append(cp.real(A[:M,:G*N] @ r + A[:M, G*N+2*M:]*x) >= b[:M])
        constraints.append(A[M:, :G*N] @ r + A[M:, G*N+2*M:end] @ x == 0)
        # PSD constraints
        for g in range(G):
            constraints.append(cp.reshape(x[g*N*N:(g+1)*N*N]) >> 0)
        constraints.append(cp.imag(r[[n*N for n in range(G)]]))

        prob = cp.Problem(obj, constraints)
        prob.solve()


        # return the W matrix, z vector, objective value and optimality status
        if prob.status in ['infeasible', 'unbounded']:
            print('infeasible solution') 
            return None, None, np.inf, True

        return None, None, np.inf, True
        

    @staticmethod
    def v_theta(theta, N):
        """
        Create vandermonde channels using the angles theta
        """
        powers = np.arange(N)
        powers = np.expand_dims(powers, 1)
        theta1 = np.expand_dims(theta.copy(), 0)
        H = np.exp(1j*theta1*powers)
        return H
        
    def generate_vandermonde_channels(self, random=True):
        pass