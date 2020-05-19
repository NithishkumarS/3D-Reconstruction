import numpy as np
from scipy import optimize

def reprojection_error(x2d, X, poses):
    X = np.concatenate([X,np.array([1])])

    error = 0
    for i in range(2):
        x = x2d[i]
        P = poses[i]
        error += (x[0]- np.dot(P[0],X) / np.dot(P[2], X))**2 + (x[1]- np.dot(P[1],X) / np.dot(P[2], X))**2
    
    return  error

def loss_func(X0, P, x1,x2):
    diff = []
    P1 = P[0]
    P2 = P[1]

    X0 = X0.reshape((-1,3))

    for _x1,_x2, _X in zip(x1,x2, X0):
        u1 = _x1[0]
        v1 = _x1[1]
        u2 = _x2[0]
        v2 = _x2[1]

        X = np.concatenate([_X, np.array([1])])

        diff.append(reprojection_error([_x1,_x2], _X, P))
    return np.array(diff).flatten()

def NonLinearTraingualtion(R2, C2, k, x1, x2, Points3D,R1=np.eye(3),C1=np.zeros((3,1))):

    P1 = np.dot(k, np.dot(R1,np.hstack((np.eye(3), -C1))))
    P2 = np.dot(k, np.dot(R2,np.hstack((np.eye(3), -C2))))
    X0 = Points3D[:,:3]
    X0 = X0.reshape((-1))
    refined_params = optimize.least_squares(loss_func, x0=X0, args=[[P1, P2], x1, x2], max_nfev=10)
    points3D = refined_params.x
    points3D = points3D.reshape((-1,3))

    return points3D
