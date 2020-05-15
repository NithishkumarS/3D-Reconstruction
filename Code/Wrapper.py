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
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
# import utils.utils as ut
# from features import *
import copy
import itertools
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *
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

def visualize(world_pts, name, file= 'save.png'):
    plt.figure()
    plt.title(name)

    for X in world_pts:
        x = X[0]
        y = X[2]
        plt.plot(x, y, 'b+')
    plt.savefig(file)
    # plt.show()


def viz_3D(tripoints3d):
    print(tripoints3d[0])
    # dgf
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.plot(tripoints3d[:,0], tripoints3d[:,1], tripoints3d[:,2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=45, azim=40)
    plt.show()

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

    data = getInliersRANSAC(iterations, images)
    print('got inliers ')

    for key, info in data.items():
        F = info[0]
        inliers = info[1]
        matches = info[2]
        cv2.imshow('matches', cv2.resize(matches, (0, 0), fx=0.5, fy=0.5))
        E = getEssentialMatrix(F, getk())
        poses = ExtractCameraPose(E)
        print('Poses computed')

        points3D = LinearTriangulation(poses, inliers, getk())
        # visualize(points3D)
        # import pdb
        # pdb.set_trace()
        bestPose, points3D = DisambiguateCameraPose(poses, points3D)
        viz_3D(points3D)

        visualize(points3D, 'linear', file= "output/"+str(key)+".png")
        print('Non linear triangulation')
        # points = NonLinearTraingualtion(bestPose[0], bestPose[1], getk(), inliers[0], inliers[1], points3D)
        # visualize(points, 'non linear',file= "output/"+"NT"+str(key)+".png")
        # print(np.mean(abs(points3D - points)))
        # print('done')


        # R,C = PnpRANSAC(inliers, points3D,getk())
        cv2.waitKey(1)
        # plt.show()

if __name__ == '__main__':
    main()

