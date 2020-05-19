import numpy as np
import random
from LinearPnP import *

def reprojection_error(x, X, P):
    X = np.concatenate([X,np.array([1])])
    # print(x)
    # print(X)
    # print(zach)
    x=x[0]
    error = (x[0]- np.dot(P[0],X) / np.dot(P[2], X))**2 + (x[1]- np.dot(P[1],X) / np.dot(P[2], X))**2
    # print(error)
    return  error

def PnpRANSAC(img_pts,world_pts, K):
    n = 0
    S_inliers = []
    min = np.inf
    max = 0
    for iter in range(5000):
        rand_idx = random.sample(range(len(img_pts)), k=6)
        # print(np.array(img_pts)[rand_idx])
        R , C , P= LinearPnP(np.array(img_pts)[rand_idx],np.array(world_pts)[rand_idx], K)
        # print(np.shape(R))
        # print(np.shape(-np.dot(R,C)))
        P = np.dot(K,np.hstack((R, -np.dot(R,C[:,np.newaxis]))))
        S = []

        for j in range(len(world_pts)):
            # val = reprojection_error(img_pts[j], world_pts[j], P)
            # print(val)
            # if val < min:
            #     min = val
            # if val > max:
            #     max = val
            # print(reprojection_error(img_pts[j], world_pts[j], P))
            if reprojection_error(img_pts[j], world_pts[j], P) < 400:
                S.append(j)
            # print(zach)
            if n < len(S):
                n = len(S)
                S_inliers = S
                # S_points_inliers = S_points
    # print(min , max)
    # dd
    x1 = []
    X1 = []

    for r in range(len(S_inliers)):#random.sample(range(len(S_inliers)), k=6):
        x1.append(img_pts[S_inliers[r]])
        X1.append(world_pts[S_inliers[r]])
    print(100.0*float(len(S_inliers))/len(world_pts),'%')
    R,C,P = LinearPnP(x1,X1, K)

    return R,C