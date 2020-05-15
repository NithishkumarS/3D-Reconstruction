import numpy as np
from scipy import optimize

def reprojection_error(x2d, X, poses):
    X = np.concatenate([X,np.array([1])])
    # print(P[0])
    # print(X)
    error = 0
    for i in range(2):
        x = x2d[i]
        P = poses[i]
    error += (x[0]- np.dot(P[0],X) / np.dot(P[2], X))**2 + (x[1]- np.dot(P[1],X) / np.dot(P[2], X))**2
    # print(error)
    
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
        # error = (x[0] - np.dot(P[0], X) / np.dot(P[2], X)) ** 2 + (x[1] - np.dot(P[1], X) / np.dot(P[2], X)) ** 2
        # print(error)
        # diff.append(  (u1 - np.dot(P1[0], X) / np.dot(P1[2], X)) ** 2 + (v1 - np.dot(P1[1], X) / np.dot(P1[2], X)) ** 2)

        diff.append(reprojection_error([_x1,_x2], _X, P))

    # X0 = X0.reshape((-1, 1))
    # print(diff)
    # print(np.shape(np.array(diff).flatten()))
    return np.array(diff).flatten()

def NonLinearTraingualtion(R, C, k, x1, x2, Points3D):

    P1 = np.dot(k, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(k, np.dot(R,np.hstack((np.eye(3), -C))))
    # P2 = np.dot(k,np.hstack(R,-C))
    # print(Points3D[0])
    # print(np.shape(Points3D))
    X0 = Points3D[:,:3]
    # print(X0)
    X0 = X0.reshape((-1))
    # print(X0.shape)

    # import pdb
    # pdb.set_trace()
    # params = [A[0, 0], A[1, 1], A[0, 2], A[1, 2], A[0, 1], k1, k2]
    refined_params = optimize.least_squares(loss_func, x0=X0, args=[[P1, P2], x1, x2])
    # print(refined_params)
    points3D = refined_params.x
    points3D = points3D.reshape((-1,3))

    return points3D
