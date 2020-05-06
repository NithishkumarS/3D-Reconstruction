import numpy as np
import random
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# print(cv2.__version__)
import fileinput, optparse
# from matplotlib import pyplot as plt
from EstimateFundamentalMatrix import *

def getMatches(img1, img2):
    # Initiate SIFT detector

    # img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # orb = cv2.ORB()
    # # create BFMatcher object
    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = orb.detectAndCompute(img1_g, None)
    # zkcn
    #
    # kp2, des2 = orb.detectAndCompute(img2_g, None)
    #
    #
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #
    # # Match descriptors.
    # matches = bf.match(des1, des2)
    #
    #
    # # Sort them in the order of their distance.
    # matches = sorted(matches, key=lambda x: x.distance)
    #
    # # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)
    #
    # plt.imshow(img3), plt.show()
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT()
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_g, None)
    kp2, des2 = sift.detectAndCompute(img2_g, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2)

    # plt.imshow(img3), plt.show()

    return matches

def importMatches():
    Matches = {}
    for c in range(1,6):
        with open('../Data/matching'+str(c)+'.txt','r') as f_open:
            data = f_open.readline()
            nFeaturers = int(data[11:])
            for i in range(nFeaturers):
                data = f_open.readline()
                sections = data.split(' ')
                nMatches = int(sections[0])
                p1 = [float(sections[4]),float(sections[5])]
                for j in range(nMatches-1):
                    index = 6+(j*3)
                    imgRef = sections[index]
                    key = str(c)+imgRef
                    p2 = [float(sections[index+1]),float(sections[index+2])]
                    if key in Matches:
                        x = Matches[key]
                        x.append([p1,p2])
                        Matches[key] = x
                    else:
                        Matches[key] = [[p1,p2]]
    return Matches
    
def getInliersRANSAC(M):

    # matches = getMatches(img1,img2)
    Matches = importMatches()
    Data = {}
    episilon = 10
    for key,matches  in Matches.items():
        c1 = np.hstack((np.array(matches)[:,0], np.ones((len(matches),1))) )
        c2 = np.hstack((np.array(matches)[:,1], np.ones((len(matches),1))) )

        S_inliers = []
        S_points_inliers = []
        n = 0
        for i in range(M):
            rand_idx = random.sample(range(len(c2)), k=8)
            F = computeFundamentalMatrix(c1[rand_idx], c2[rand_idx])
            S = []
            S_points = []
            for j in range(len(c1)):
                x1, x2  = c1[j],c2[j]
                if np.dot(np.dot(x2.T, F),x1) < episilon:
                    S.append(j)
                    S_points.append([x1[:2],x1[:2]])

                if n <len(S):
                    n = len(S)
                    S_inliers = S
                    S_points_inliers = S_points
        X1 = []
        X2 = []

        for r in random.sample(range(len(S)), k=8):
            X1.append(c1[S_inliers[r]])
            X2.append(c2[S_inliers[r]])
        F= computeFundamentalMatrix(X1, X2)
        Data[key] = [F,S_points_inliers]
    return Data

# importMatches()
# b = getInliersRANSAC(cv2.imread('../Data/1.jpg'),cv2.imread('../Data/2.jpg') , 1)
