from cvxopt import matrix, solvers
import numpy as np


def solve_qp_relaxed(H, l=None, u=None):
    # min  w'w
    # s.t. |H(:,j)'w|>=1, j=1,...,m
    # We assume Image(w_n)=0;

    _, n, m = H.shape
    dim = 2*n

    tcons = ( (u-l)<= 2*np.pi-0.01 )
    tcons[0] = False
    ncons = sum(tcons) # number of inequality constraints l_m <= |c_m| <= u_m
    nfree = ncons*2 
    nvar = nfree + dim # size of w(r and i) and c with ineq constraints 
    neq = ncons*2 + 1 # number of equality constraint c_k = h_k^H w (real and imag), 
    nieq = ncons*3 + 1 # number of inequalit constraints in sin cos form
    Q = np.eye(nvar)*2 
    Q[:nfree, :nfree] = 0

    c = np.zeros((nvar,1))

    A = np.zeros((nieq, nvar))
    b = np.zeros((nieq, 1))
    Aeq = np.zeros((neq,nvar))
    beq = np.zeros((neq,1))
    # print(ncons, nfree, nvar, neq, nieq)

    id=0
    ix=0
    for t in range(1,m,1):
        if tcons[t] == 0:
            continue
        
        hr = H[0,:,t]    # 0 index is for real and 1 for imaginary
        hi = H[1,:,t]
        
        Aeq[ix, id*2] = -1
        Aeq[ix, nfree:] = np.concatenate((hr, hi))
        
        ix = ix+1
        Aeq[ix, id*2+1] = -1
        Aeq[ix, nfree:] = np.concatenate((-hi, hr))

        ix = ix + 1
        id = id + 1

    hr = H[0,:,0]
    hi = H[1,:,0]

    Aeq[ix, nfree:nvar] = np.concatenate((-hi, hr))

    t = -1
    ix = 0
    for j in range(1,m,1):
        if  tcons[j]:
            t = t + 1
        else:
            continue
        
        x1 = np.cos(l[j])
        y1 = np.sin(l[j])

        x2 = np.cos(u[j])
        y2 = np.sin(u[j])

        x3 = np.cos((l[j] + u[j])/2)
        y3 = np.sin((l[j] + u[j])/2)

        cc = complex((y1-y2), (x1-x2))
        
        A[ix , t*2] = y1 - y2;
        A[ix , t*2+1] =  -x1 + x2;

        b[ix] = (y1*x2 - y2*x1);
        zr = ((cc* complex(x3, y3)) + y2*x1-y1*x2).real;
        if zr > 0:
            A[ix,:] = -A[ix,:]
            b[ix] = -b[ix]

        ix = ix + 1
        
        A[ix , t*2] = y1
        A[ix , t*2+1] =  -x1
        b[ix] = 0
        zr = x3*y1 - y3*x1
        if zr>0:
            A[ix,:] = -A[ix,:]
            b[ix] = -b[ix]
        
        ix = ix + 1

        A[ix , t*2] = y2
        A[ix , t*2+1]   =  -x2
        b[ix] = 0
        zr = x3*y2 - y3*x2

        if zr>0:
            A[ix,:] = -A[ix,:]
            b[ix] = -b[ix]
        
        ix = ix + 1

    A[ix, nfree:] = np.concatenate((-hr,-hi))
    b[ix] = -1

    # optnew = optimset('Display','off','LargeScale','off');


    Q = matrix(Q)
    c = matrix(c)
    A = matrix(A)
    b = matrix(b)
    Aeq = matrix(Aeq)
    beq = matrix(beq)
    # print("A", A)
    # print("b", b)
    solvers.options['show_progress'] = False
    solution = solvers.qp(Q,c,A,b,Aeq,beq)
    optimal = False
    if solution['status'] == 'optimal':
        optimal = True

    wr = np.array(solution['x'][nfree:])
    w = wr[:n] + wr[n:2*n+1]*1j
    # print(solution['y'])
    return w, np.array(solution['primal objective']), optimal


if __name__=='__main__':
    n = 3
    m = 2
    H = np.random.randn(2, n, m)

    H = np.array([[[ 1.27109919,  0.62238554],
                [-0.38933997,  0.48843181],
                [-0.14073963, -0.77918651]],

                [[-1.40115267, -0.17910412],
                [-0.82520195, -1.08825745],
                [-0.62317508, -0.67931762]]])
    l = np.zeros(m)
    u = np.ones(m)*np.pi*2

    solve_qp_relaxed(H, l, u)