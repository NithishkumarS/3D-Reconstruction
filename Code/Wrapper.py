#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project: Structure from Motion

Author(s):
Nithish Kumar

Zack

Needed Directories



"""

# Code starts here:
import argparse
import numpy as np
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
# import utils.utils as ut
# from features import *
import copy
import itertools
import random
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from LinearPnP import *
from PnPRANSAC import *
import matplotlib.pyplot as plt

def graphPoints(points):
    print(np.shape(points))

def getk():
    K = np.array([[568.996140852, 0, 643.21055941],
    [0, 568.988362396, 477.982801038],
    [0, 0, 1]] )
    return K

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--ImgDirectory', default='../Data/',
                        help='Directory that contains images for Sfm')

    # Parser.add_argument('--compute_corners', default=True, help='Directory that contains images for panorama sticking')
    Args = Parser.parse_args()
    ImgDirectory = Args.ImgDirectory
    Fundamental_matrix = {}
    num_files = len(os.listdir(ImgDirectory))
    images = []
    for i in range(1, 7):
        img = cv2.imread(ImgDirectory + str(i) + '.jpg')
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        images.append(img)

    iterations = 1000

    data = getInliersRANSAC(iterations,images)
    # np.save('data.npy',data)
    # data = np.load('data.npy')
    for key,info in data.items():
        F = info[0]
        inliers = info[1]
        matches = info[2]
        cv2.imshow('matches',cv2.resize(matches,(0,0),fx=0.5, fy=0.5))
        E = getEssentialMatrix(F,getk())
        poses = ExtractCameraPose(E)

        plt.plot(0,0,'rx')
        plt.plot(poses[0][0][0],poses[0][0][2],'bx')
        plt.plot(poses[1][0][0],poses[1][0][2],'gx')
        plt.plot(poses[2][0][0],poses[2][0][2],'cx')
        plt.plot(poses[3][0][0],poses[3][0][2],'kx')
        # plt.yaxis.label('Z')
        # plt.xaxis.label('X')


        points3D= LinearTriangulation(poses,inliers,getk())
        # plt.plot(points3D[:,0,0],points3D[:,0,2],'bo')
        # plt.plot(points3D[:,1,0],points3D[:,1,2],'go')
        # plt.plot(points3D[:,2,0],points3D[:,2,2],'co')
        # plt.plot(points3D[:,3,0],points3D[:,3,2],'ko')
        




        # print(len(inliers), len(inliers[0]), len(points3D), len(points3D[0]))
        bestPose,points3D = DisambiguateCameraPose(poses,points3D)
        plt.plot(points3D[:,0],points3D[:,2],'ro')
        # print(np.shape(points3D))
        # plt.scatter(points3D)
        # print(np.shape(points3D))
        # R,C = PnpRANSAC(inliers, points3D,getk())
        cv2.waitKey(1)
        plt.show()


if __name__ == '__main__':
    main()

