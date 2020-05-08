import numpy as np
import random
from LinearPnP import *

def reprojection_error(x, X, P):
    X = np.concatenate([X,np.array([1])])
    # print()
    x=x[0]
    error = (x[0]- np.dot(P[0],X) / np.dot(P[2], X))**2 + (x[1]- np.dot(P[1],X) / np.dot(P[2], X))**2
    # print(error)
    return  error

def PnpRANSAC(img_pts,world_pts, K):
    n = 0
    S_inliers = []
    min = np.inf
    max = 0
    for iter in range(100):
        rand_idx = random.sample(range(len(img_pts)), k=6)
        print(np.array(img_pts)[rand_idx])
        R , C , P= LinearPnP(np.array(img_pts)[rand_idx],world_pts[rand_idx], K)
        # print(R,C)

        S = []

        for j in range(len(world_pts)):
            # val = reprojection_error(img_pts[j], world_pts[j], P)
            # print(val)
            # if val < min:
            #     min = val
            # if val > max:
            #     max = val

            if reprojection_error(img_pts[j], world_pts[j], P) < 30000:
                S.append(j)

            if n < len(S):
                n = len(S)
                S_inliers = S
                S_points_inliers = S_points
    # print(min , max)
    # dd

    return R,C