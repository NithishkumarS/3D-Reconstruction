import numpy as np
import cv2

def LinearPnP(img_pts,world_pts, K):
    world_pts = np.hstack((np.array(world_pts), np.ones((len(world_pts),1)) ))

    _zero = np.zeros(4)
    A = np.empty((0,12))
    for _x, X in zip(img_pts, world_pts):
        x = _x[1]
        u = x[0]
        v = x[1]
        row = np.array([[ 0,     0,     0,     0,  -X[0],-X[1],-X[2], -X[3],  v*X[0],v*X[1],v*X[2], v*X[3] ],
                      [ X[0], X[1], X[2],   X[3],     0,     0,     0,    0,   u*X[0],u*X[1],u*X[2], u*X[3] ],
                      [-v*X[0],-v*X[1],-v*X[2], -v*X[3], u*X[0],u*X[1],u*X[2], u*X[3], 0,     0,     0,    0,  ] ])
        A = np.vstack((A,row))
    world_pts = np.array(world_pts[:,:3],dtype=np.float32)
    img_pts = np.array(np.array(img_pts)[:,1],dtype= np.float32)

    u,s,v = np.linalg.svd(A)
    P = v[-1]
    P = P.reshape((3,4))
    # Temp = cv2.decomposeProjectionMatrix(P,K)
    # Rt = Temp[1]
    # Ct = Temp[2]

    R = np.dot(np.linalg.inv(K), P)[:,0:3]
    u1, d1, v1 = np.linalg.svd(R)
    R = np.matmul(u1, v1)

    t = np.dot(np.linalg.inv(K), P)[:,3]#/d1[0]

    if np.linalg.det(R) == -1:
        # print(-1)
        R = -R
        # C = -C
    C = -np.dot(np.linalg.inv(R),t)
    # Rt = cv2.Rodrigues(Pt[1])[0]
    # Ct = Pt[2]


    

    return R, C, P
