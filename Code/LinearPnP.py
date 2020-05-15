import numpy as np

def LinearPnP(img_pts,world_pts, K):
    world_pts = np.hstack((np.array(world_pts), np.ones((len(world_pts),1)) ))

    _zero = np.zeros(4)
    A = np.empty((0,12))
    for xi, X in zip(img_pts, world_pts):
        x = xi[0]
        row = np.array([[ 0,     0,     0,     0,  -X[0],-X[1],-X[2], -X[3],  x[1]*X[0],x[1]*X[1],x[1]*X[2], x[1]*X[3] ],
                      [ X[0], X[1], X[2],   X[3],     0,     0,     0,    0,   x[0]*X[0],x[0]*X[1],x[0]*X[2], x[0]*X[3] ],
                      [-x[1]*X[0],-x[1]*X[1],-x[1]*X[2], -x[1]*X[3], x[0]*X[0],x[0]*X[1],x[0]*X[2], x[0]*X[3], 0,     0,     0,    0,  ] ])
        A = np.vstack((A,row))
        # print(count)
        # count +=1

    # print(A)
    # print(A.shape)

    u,s,v = np.linalg.svd(A)
    P = v[:,-1]/v[-1,-1]
    P = P.reshape((3,4))

    # print(P)
    # print('1:3',P[:,0:3])

    R = np.dot(np.linalg.inv(K), P)[:,0:3]
    u1, d1, v1 = np.linalg.svd(R)
    R = np.matmul(u1, v1)
    # print(R)
    # print('d1',d1)

    t = np.dot(np.linalg.inv(K), P)[:,3]/d1[0]

    if np.linalg.det(R) == -1:
        # print(-1)
        R = -R
    C = -np.dot(R,t)
    # C = np.dot(-np.linalg.inv(R.T),P[:,-1])
    # C = np.dot(np.linalg.inv(R),np.dot(np.linalg.inv(K),P))[:,-1]
    # print(R)
    # print(np.linalg.inv(R.T))
    # print('C', C)

    return R, C, P
