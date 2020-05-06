import numpy as np
import random
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

print(cv2.__version__)
# from matplotlib import pyplot as plt
# from EstimateFundamentalMatrix import *

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

def getInliersRANSAC(img1, img2, M):

    matches = getMatches(img1,img2)

    c1 = np.hstack((np.array(matches[0]), np.ones((len(c1),1))) )
    c2 = np.hstack((np.array(matches[1]), np.ones((len(c2),1))) )
    S_inliers = []
    n = 0
    for i in range(M):

        rand_idx = sample(range(len(c2)), k=8)
        F = computeFundamentalMatrix(c1[rand_idx], c2[rand_idx])
        S = []
        for j in range(len(c1)):
            x1, x2  = c1[j],c2[j]
            if np.linalg.det( np.dot(np.dot(x2.T, F),x1) ) < episilon:
                S.append(j)

            if n <len(S):
                n = len(S)
                S_inliers = S
    return F,S_inliers
a = getMatches(cv2.imread('../Data/1.jpg'),cv2.imread('../Data/2.jpg') )