from cvxopt import matrix, solvers
import numpy as np
import cvxpy as cp
import time

epsilon = 0.00001

# def check_feasibility(H=None, l=None, u=None, w=None, z=None, T=None):
#     Hc = H[0,::] + 1j*H[1,::]
#     if T is not None:
#         theta = np.angle(np.matmul(Hc.conj().T, w))
#         z_feasible = (np.abs(np.real(w)) <= T*z + epsilon).all() and (np.abs(np.imag(w)) <= T*z + epsilon).all()
#     else:
#         theta = np.angle(np.matmul(Hc.conj().T, w*z))
#         z_feasible = True
#     theta[theta<-epsilon] += 2*np.pi
#     theta_feasible = (theta <= u + epsilon).all() and (theta >= l - epsilon).all()
#     return theta_feasible and z_feasible

def check_feasibility(H=None, l=None, u=None, w=None, z=None, T=None):
    Hc = H[0,::] + 1j*H[1,::]
    if T is not None:
        theta = np.angle(np.matmul(Hc.conj().T, w))
        z_feasible = (np.abs(np.real(w)) <= T*z + epsilon).all() and (np.abs(np.imag(w)) <= T*z + epsilon).all()
        # print('z', z_feasible)
    else:
        theta = np.angle(np.matmul(Hc.conj().T, w))
        z_feasible = True
    theta[theta<-epsilon] += 2*np.pi
    # print('theta', theta)
    theta_feasible = (((theta + 2*np.pi <= u + epsilon) & (theta + 2*np.pi >= l-epsilon)) | ((theta <= u + epsilon) & (theta >= l - epsilon))).all()
    # print(theta_feasible)
    
    return theta_feasible and z_feasible

def check_integrality(z_sol, z_mask):
    sum_z= np.sum(np.abs(z_mask*(z_sol - np.round(z_sol))))
    return sum_z < 0.001

def qp_relaxed(H, l=None, u=None, z_mask=None, z_sol=None, max_ant=None, T=1000):
    # min  w'w
    # s.t. |H(:,m)'w|>=1, m=1,...,M
    # We assume Image(w_n)=0;
    # print(H, l,u,z_mask, z_sol, max_ant)

    assert l is not None and u is not None and z_mask is not None and z_sol is not None and max_ant is not None, "One of the arguments is None"
    assert check_integrality(z_sol, z_mask), "Solving relaxed: selected antennas part not integral"

    _, N, M = H.shape # numeber of antennas and users, resp.
    twoN = 2*N

    mask_constr = ( (u-l)<= 2*np.pi-0.01 )
    # print('mask constraints', mask_constr)
    mask_constr[0] = False
    num_constr = sum(mask_constr) # number of inum_equality constraints l_m <= |c_m| <= u_m
    
    num_z_eq_constr = int(sum(z_mask)) + 1
    num_z_ineq_constr = N*6 + 1  # z(n)>=0, z(n)<=1, w(n)<=z(n)*M, w(n)>=-z(n)*M, z^T 1 <=L 
    
    c_dim = num_constr*2 # vector of stacked real part and imaginary part of c variable, i.e. [real(c) imag(c)]
    cw_dim = c_dim + twoN # size of w(r and i) and c with inum_eq constraints 
    cwz_dim = c_dim + twoN + N

    num_c_eq = num_constr*2 + 1
    num_eq = num_constr*2 + 1 + num_z_eq_constr # number of equality constraint c_k = h_k^H w (real and imag), 
    
    num_c_ineq = num_constr*3 + 1  
    num_ineq = num_constr*3 + num_z_ineq_constr # number of inum_equalit constraints in sin cos form
    Q = np.eye(cwz_dim)*2 

    Q[:c_dim, :c_dim] = 0 # since c does not appear in the objective
    Q[cw_dim:, cw_dim:] = 0 # since z does not appear in the objective

    c = np.zeros((cwz_dim,1)) # there is not linear term in the objective

    # inequality parameters
    A = np.zeros((num_ineq, cwz_dim)) 
    b = np.zeros((num_ineq, 1))

    # Equality parameters
    Aeq = np.zeros((num_eq, cwz_dim))
    beq = np.zeros((num_eq, 1))

    y_ind=0
    x_ind=0

    for m in range(1,M,1):
        if mask_constr[m] == 0:
            continue
        
        hm_real = H[0,:,m]    # 0 index is for real and 1 for imaginary
        hm_imag = H[1,:,m]
        
        Aeq[x_ind, y_ind*2] = -1  # to push c behind the equality (for the real part)
        Aeq[x_ind, c_dim:cw_dim] = np.concatenate((hm_real, hm_imag))
        
        x_ind = x_ind+1
        Aeq[x_ind, y_ind*2+1] = -1  # to push c behind the equality (for the imag part)
        Aeq[x_ind, c_dim:cw_dim] = np.concatenate((-hm_imag, hm_real))

        x_ind = x_ind + 1
        y_ind = y_ind + 1

    hm_real = H[0,:,0]
    hm_imag = H[1,:,0]

    Aeq[x_ind, c_dim:cw_dim] = np.concatenate((-hm_imag, hm_real)) # for the first element, imag(h1^H w) = 0

    # z related equality constraints:
    x_ind = num_c_eq
    for n in range(N):
        if not z_mask[n]:
            continue

        Aeq[x_ind, cw_dim+n] = 1
        beq[x_ind] = z_sol[n]
        x_ind += 1
        
    # z^T 1 == L
    Aeq[x_ind, cw_dim:cwz_dim] = 1
    beq[x_ind] = max_ant
    x_ind += 1


    t = -1
    x_ind = 0
    for m in range(1,M,1):
        if  mask_constr[m]:
            t = t + 1
        else:
            continue
        
        lx = np.cos(l[m])
        ly = np.sin(l[m])

        ux = np.cos(u[m])
        uy = np.sin(u[m])

        midx = np.cos((l[m] + u[m])/2)
        midy = np.sin((l[m] + u[m])/2)

        cc = complex((ly-uy), (lx-ux))
        
        A[x_ind , t*2] = ly - uy
        A[x_ind , t*2+1] =  -lx + ux

        b[x_ind] = (ly*ux - uy*lx)
        zr = ((cc* complex(midx, midy)) + uy*lx-ly*ux).real
        if zr > 0:
            A[x_ind,:] = -A[x_ind,:]
            b[x_ind] = -b[x_ind]

        x_ind = x_ind + 1
        
        A[x_ind , t*2] = ly
        A[x_ind , t*2+1] =  -lx
        b[x_ind] = 0
        zr = midx*ly - midy*lx
        if zr>0:
            A[x_ind,:] = -A[x_ind,:]
            b[x_ind] = -b[x_ind]
        
        x_ind = x_ind + 1

        A[x_ind , t*2] = uy
        A[x_ind , t*2+1]   =  -ux
        b[x_ind] = 0
        zr = midx*uy - midy*ux

        if zr>0:
            A[x_ind,:] = -A[x_ind,:]
            b[x_ind] = -b[x_ind]
        
        x_ind = x_ind + 1


    A[x_ind, c_dim:cw_dim] = np.concatenate((-hm_real,-hm_imag))
    b[x_ind] = -1

    # inequality constraints related to z variable
    x_ind = num_c_ineq
    for n in range(N):
        # z(n) >= 0
        # print('x_ind {}, y_ind {}'.format(x_ind, n+cw_dim))
        A[x_ind, n+cw_dim] = -1
        x_ind += 1
        
        # z(n) <= 1
        A[x_ind, n+cw_dim] = 1
        b[x_ind] = 1
        x_ind += 1
        
        # w(n) <= z(n)*M
        A[x_ind, n+c_dim] = 1    # real(w(n))
        A[x_ind, n+cw_dim] = -T
        x_ind += 1
        
        # w(n) <= z(n)*M
        A[x_ind, n+N+c_dim] = 1  # imag(w(n))
        A[x_ind, n+cw_dim] = -T
        x_ind += 1

        # w(n) >= -z(n)*M
        A[x_ind, n+c_dim] = -1   # real(w(n))
        A[x_ind, n+cw_dim] = -T
        x_ind += 1
        
        # w(n) >= -z(n)*M
        A[x_ind, n+N+c_dim] = -1 # imag(w(n))
        A[x_ind, n+cw_dim] = -T
        x_ind += 1

    # z^T 1 <= L
    # A[x_ind, cw_dim:cwz_dim] = 1
    # b[x_ind] = max_ant
    # x_ind += 1

    # optnew = optimset('Display','off','LargeScale','off');

    Q   = matrix(Q)
    c   = matrix(c)
    A   = matrix(A)
    b   = matrix(b)
    Aeq = matrix(Aeq)
    beq = matrix(beq)
    # print("A", A)
    # print("b", b)
    solvers.options['show_progress'] = False

    # solution = solvers.qp(Q,c,A,b,Aeq,beq)

    try:
        solution = solvers.qp(Q,c,A,b,Aeq,beq)
    except:
        return cvxpy_relaxed(H,l=l,u=u, z_mask=z_mask, z_sol=z_sol, max_ant=max_ant, T=T)

    wz = np.array(solution['x'][c_dim:])
    w = wz[:N] + wz[N:2*N]*1j
    z = wz[2*N:]

    optimal = False
    is_feasible = False
    if solution['status'] == 'optimal':
        # print('feasible')

        optimal = True
        is_feasible = True # automatically holds
    else:
        # print('checking feasibility')
        # check feasibility
        is_feasible = check_feasibility(H=H, l=l, u=u, w=w.squeeze(), z=z.squeeze(), T=T)
        # print('checking feasibility', is_feasible)

        if is_feasible:
            z_cp, w_cp, obj_cp, opt_cp = cvxpy_relaxed(H,l=l,u=u, z_mask=z_mask, z_sol=z_sol, max_ant=max_ant, T=T)
            # print('cvxpy result', opt_cp)
            if opt_cp:
                w = w_cp.copy()
                z = z_cp.copy()
            optimal = opt_cp 

    # print(solution['y'])

    # return z.squeeze(), w.squeeze(), np.array(solution['primal objective']), optimal
    return z.squeeze(), w.squeeze(), np.linalg.norm(w.squeeze(), 2)**2, optimal

def qp_relaxed_with_selected_antennas(H, l=None, u=None, z_sol=None, M=1000):

    _, N, M = H.shape # numeber of antennas and users, resp.
    assert sum(z_sol) >= 1, "No antenna selected"
    assert check_integrality(z_sol, np.ones(N)), "Solving relaxed: selected antennas part not integral"

    twoN = 2*N

    mask_constr = ( (u-l)<= 2*np.pi-0.01 )
    mask_constr[0] = False
    num_constr = sum(mask_constr) # number of inum_equality constraints l_m <= |c_m| <= u_m
    
    c_dim = num_constr*2 # vector of stacked real part and imaginary part of c variable, i.e. [real(c) imag(c)]
    cw_dim = c_dim + twoN # size of w(r and i) and c with inum_eq constraints 

    num_eq = num_constr*2 + 1 # number of equality constraint c_k = h_k^H w (real and imag), 
    
    num_ineq = num_constr*3 + 1  # number of inum_equalit constraints in sin cos form
    
    Q = np.eye(cw_dim)*2 

    # mask the objective
    # for i in range(N):
    #     if not z_sol[i]:
    #         Q[c_dim+i,c_dim+i] = 0
    #         Q[c_dim+N+i,c_dim+N+i] = 0        


    Q[:c_dim, :c_dim] = 0 # since c does not appear in the objective

    c = np.zeros((cw_dim,1)) # there is not linear term in the objective

    # inequality parameters
    A = np.zeros((num_ineq, cw_dim)) 
    b = np.zeros((num_ineq, 1))

    # Equality parameters
    Aeq = np.zeros((num_eq, cw_dim))
    beq = np.zeros((num_eq, 1))

    y_ind=0
    x_ind=0

    for m in range(1,M,1):
        if mask_constr[m] == 0:
            continue
        
        hm_real = H[0,:,m]*z_sol.squeeze()    # 0 index is for real and 1 for imaginary
        hm_imag = H[1,:,m]*z_sol.squeeze()
        
        Aeq[x_ind, y_ind*2] = -1  # to push c behind the equality (for the real part)
        Aeq[x_ind, c_dim:cw_dim] = np.concatenate((hm_real, hm_imag))
        
        x_ind = x_ind+1
        Aeq[x_ind, y_ind*2+1] = -1  # to push c behind the equality (for the imag part)
        Aeq[x_ind, c_dim:cw_dim] = np.concatenate((-hm_imag, hm_real))

        x_ind = x_ind + 1
        y_ind = y_ind + 1

    # print('shapes', z_sol.shape, H[0,:,0].shape)
    hm_real = H[0,:,0]*z_sol.squeeze()
    hm_imag = H[1,:,0]*z_sol.squeeze()

    Aeq[x_ind, c_dim:cw_dim] = np.concatenate((-hm_imag, hm_real)) # for the first element, imag(h1^H w) = 0


    t = -1
    x_ind = 0
    for m in range(1,M,1):
        if  mask_constr[m]:
            t = t + 1
        else:
            continue
        
        lx = np.cos(l[m])
        ly = np.sin(l[m])

        ux = np.cos(u[m])
        uy = np.sin(u[m])

        midx = np.cos((l[m] + u[m])/2)
        midy = np.sin((l[m] + u[m])/2)

        cc = complex((ly-uy), (lx-ux))
        
        A[x_ind , t*2] = ly - uy
        A[x_ind , t*2+1] =  -lx + ux

        b[x_ind] = (ly*ux - uy*lx)
        zr = ((cc* complex(midx, midy)) + uy*lx-ly*ux).real
        if zr > 0:
            A[x_ind,:] = -A[x_ind,:]
            b[x_ind] = -b[x_ind]

        x_ind = x_ind + 1
        
        A[x_ind , t*2] = ly
        A[x_ind , t*2+1] =  -lx
        b[x_ind] = 0
        zr = midx*ly - midy*lx
        if zr>0:
            A[x_ind,:] = -A[x_ind,:]
            b[x_ind] = -b[x_ind]
        
        x_ind = x_ind + 1

        A[x_ind , t*2] = uy
        A[x_ind , t*2+1]   =  -ux
        b[x_ind] = 0
        zr = midx*uy - midy*ux

        if zr>0:
            A[x_ind,:] = -A[x_ind,:]
            b[x_ind] = -b[x_ind]
        
        x_ind = x_ind + 1


    A[x_ind, c_dim:cw_dim] = np.concatenate((-hm_real,-hm_imag))
    b[x_ind] = -1

    Q   = matrix(Q)
    c   = matrix(c)
    A   = matrix(A)
    b   = matrix(b)
    Aeq = matrix(Aeq)
    beq = matrix(beq)
    # print("A", A)
    # print("b", b)
    solvers.options['show_progress'] = False
    try:
        solution = solvers.qp(Q,c,A,b,Aeq,beq)
    except:
        return cvxpy_relaxed_with_selected_antennas(H,l=l,u=u, z_sol=z_sol.squeeze())
    wz = np.array(solution['x'][c_dim:])
    w = wz[:N] + wz[N:2*N]*1j

    optimal = False
    is_feasible = False
    if solution['status'] == 'optimal':
        optimal = True
        is_feasible = True # automatically holds
    else:
        # check feasibility
        is_feasible = check_feasibility(H=H, l=l, u=u, w=w.squeeze(), z=z_sol.squeeze(), T=None)
        if is_feasible:
            try:
                w_cp, obj_cp, opt_cp = cvxpy_relaxed_with_selected_antennas(H,l=l,u=u, z_sol=z_sol.squeeze())
                if opt_cp:
                    w = w_cp.copy()
                optimal = opt_cp 
            except:
                optimal = is_feasible
    # z = wz[2*N:]
    # print(solution['y'])
    # return w.squeeze(),  np.array(solution['primal objective']), optimal
    return w.squeeze(),  np.linalg.norm(w.squeeze(), 2)**2, optimal


def cvxpy_relaxed(H, l=None, u=None, z_mask=None, z_sol=None, max_ant=None, T=1000):
    # l of shape M-1, u of shape M-1
    _,N,M = H.shape
    
    H = H[0,::] + 1j*H[1,::]
    w = cp.Variable(N, complex=True)
    c = cp.Variable(M-1, complex=True)
    z = cp.Variable(N)
    
    obj = cp.Minimize(cp.square(cp.norm(w, 2)))
    constraints = []

    # Equality Constraints
    constraints += [c == H[:,1:].conj().T @ w]
    
    # Inequality Constraints
    constraints += [cp.real(H[:,0].conj().T @ w) >= 1]
    constraints += [cp.imag(H[:,0].conj().T @ w) == 0]
    
    l = l[1:].copy()
    u = u[1:].copy()
    mask = (u-l)<=np.pi
    # print(mask)
    constraints += [cp.multiply(np.sin(l)*mask, cp.real(c)) - cp.multiply(np.cos(l)*mask, cp.imag(c)) <= np.zeros(M-1)]
    constraints += [cp.multiply(np.sin(u)*mask, cp.real(c)) - cp.multiply(np.cos(u)*mask, cp.imag(c)) >= np.zeros(M-1)]
    
    constraints += [cp.real(w[:]) <=  T*z]
    constraints += [cp.real(w[:]) >= -T*z]
    constraints += [cp.imag(w[:]) <=  T*z]
    constraints += [cp.imag(w[:]) >= -T*z]
    
    # one = np.ones(N)
    # constraints += [cp.sum(z)<=max_ant]

    # Enforce maximum antenna constraint
    constraints += [cp.sum(z) == max_ant]

    for n in range(N):
        if z_mask[n]:
            constraints += [z[n] == z_sol[n]]
    
    constraints += [z >= np.zeros(N), z <= np.ones(N)]
    
    a = (np.cos(l) + np.cos(u))/2 
    b = (np.sin(l) + np.sin(u))/2
    constraints += [cp.multiply(cp.multiply(a,cp.real(c)) + cp.multiply(b, cp.imag(c)), mask) >= cp.multiply(a**2 +b**2,mask)]

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(verbose=False)
    except:
        return np.zeros(N), np.zeros(N), np.inf, False
        
    optimal = True
    if w.value is None:
        optimal = False
        return np.zeros(N), np.zeros(N), np.inf, optimal
    return z.value.squeeze(), w.value.squeeze(), np.linalg.norm(w.value.squeeze(),2)**2, optimal


def cvxpy_relaxed_with_selected_antennas(H, l=None, u=None, z_sol=None, T=1000):
    """
    cvxpy implementation of qp_relaxed_with_selected_antennas
    """
    # l of shape M-1, u of shape M-1
    _,N,M = H.shape
    
    H = H[0,::] + 1j*H[1,::]
    w = cp.Variable(N, complex=True)
    c = cp.Variable(M-1, complex=True)
    
    obj = cp.Minimize(cp.square(cp.norm(w, 2)))
    constraints = []

    # Equality Constraints
    constraints += [c == H[:,1:].conj().T @ cp.multiply(w, z_sol)]
    
    # Inequality Constraints
    constraints += [cp.real(H[:,0].conj().T @ cp.multiply(w, z_sol)) >= 1]
    constraints += [cp.imag(H[:,0].conj().T @ cp.multiply(w, z_sol)) == 0]
    
    l = l[1:].copy()
    u = u[1:].copy()
    mask = (u-l)<=np.pi

    constraints += [cp.multiply(np.sin(l)*mask, cp.real(c)) - cp.multiply(np.cos(l)*mask, cp.imag(c)) <= np.zeros(M-1)]
    constraints += [cp.multiply(np.sin(u)*mask, cp.real(c)) - cp.multiply(np.cos(u)*mask, cp.imag(c)) >= np.zeros(M-1)]
    
    a = (np.cos(l) + np.cos(u))/2 
    b = (np.sin(l) + np.sin(u))/2
    constraints += [cp.multiply(cp.multiply(a,cp.real(c)) + cp.multiply(b, cp.imag(c)), mask) >= cp.multiply(a**2 +b**2,mask)]

    prob = cp.Problem(obj, constraints)

    try:
        prob.solve(verbose=False)
    except:
        return np.zeros(N), np.inf, optimal

    optimal = True
    if w.value is None:
        optimal = False
        return np.zeros(N), np.inf, optimal
    return w.value.squeeze(), np.linalg.norm(w.value.squeeze(),2)**2, optimal


# if __name__=='__main__':
#     N = 12
#     M = 6
#     H = np.random.randn(2, N, M)

#     z_sol = np.random.binomial(size=N, n=1, p= 0.5)
#     z_mask = np.random.binomial(size=N, n=1, p=0.6)

#     # H = np.array([[[ 1.27109919,  0.62238554],
#     #             [-0.38933997,  0.48843181],
#     #             [-0.14073963, -0.77918651]],

#     #             [[-1.40115267, -0.17910412],
#     #             [-0.82520195, -1.08825745],
#     #             [-0.62317508, -0.67931762]]])
#     l = np.zeros(M)
#     u = np.random.rand(M)*np.pi
#     import time
#     t1 = time.time()
#     w,z,obj,opt = solve_relaxed(H, l, u, z_mask, z_sol)
#     print('time taken {}'.format(time.time()-t1))
#     print(w,z,obj)