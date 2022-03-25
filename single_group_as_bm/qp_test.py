def solve_relaxed_with_fixed_antennas(H, l, u, z_sol, M=1000):

    _, N, M = H.shape # numeber of antennas and users, resp.
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
    for i in range(N):
        if not z_sol[i]:
            Q[c_dim+i,c_dim+i] = 0
            Q[c_dim+N+i,c_dim+N+i] = 0        


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
        
        hm_real = H[0,:,m]*z_sol    # 0 index is for real and 1 for imaginary
        hm_imag = H[1,:,m]*z_sol
        
        Aeq[x_ind, y_ind*2] = -1  # to push c behind the equality (for the real part)
        Aeq[x_ind, c_dim:cw_dim] = np.concatenate((hm_real, hm_imag))
        
        x_ind = x_ind+1
        Aeq[x_ind, y_ind*2+1] = -1  # to push c behind the equality (for the imag part)
        Aeq[x_ind, c_dim:cw_dim] = np.concatenate((-hm_imag, hm_real))

        x_ind = x_ind + 1
        y_ind = y_ind + 1

    hm_real = H[0,:,0]*z_sol
    hm_imag = H[1,:,0]*z_sol

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
    solution = solvers.qp(Q,c,A,b,Aeq,beq)
    optimal = False
    if solution['status'] == 'optimal':
        optimal = True

    wz = np.array(solution['x'][c_dim:])
    w = wz[:N] + wz[N:2*N]*1j
    # z = wz[2*N:]
    # print(solution['y'])
    return w,  np.array(solution['primal objective']), optimal
