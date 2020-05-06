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

    iterations = 100

    data = getInliersRANSAC(iterations)
    for key,info in data.items():
        F = info[0]
        inliers = info[1]
        E = getEssentialMatrix(F,getk())
        poses = ExtractCameraPose(E)
        points3D= LinearTriangulation(poses,inliers)
        bestPose,points3D = DisambiguateCameraPose(poses,points3D)
        print(np.shape(points3D))

if __name__ == '__main__':
    main()

