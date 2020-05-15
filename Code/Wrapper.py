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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *
from LinearPnP import *
from PnPRANSAC import *
import matplotlib.pyplot as plt
import pickle

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
def rotmatrix_to_angles(M):
    r = R.from_matrix(M)
    print(r.as_euler('zyx', degrees=True))

def viz_3D(tripoints3d):
    # print(tripoints3d[0])
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

def plotCamera(R,C,color):

    pnts = np.array([[0,0,0],[-1,0,1],[1,0,1],[0,0,0]])
    plt.plot(pnts[:,0],pnts[:,2],'k')
    plt.plot(pnts[0,0],pnts[0,2],'ko')
    pnts = np.dot(pnts,R)
    pnts[:,0] = pnts[:,0]+C[0]
    pnts[:,2] = pnts[:,2]+C[2]  
    plt.plot(pnts[:,0],pnts[:,2],color)
    plt.plot(pnts[0,0],pnts[0,2],'ko')

    return plt

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

    iterations = 5000

    # data = getInliersRANSAC(iterations, images)
    # handle =  open('data.pkl','wb')
    # pickle.dump(data,handle)
    # handle.close()
    
    pickleFile = open('data.pkl','rb')
    data = pickle.load(pickleFile)
    pickleFile.close()

    print('got inliers ')

    for key, info in data.items():
    	info = data["12"]
    	key = "12"
        F = info[0]
        inliers = np.array(info[1])
        matches = info[2]
        cv2.imshow('matches', cv2.resize(matches, (0, 0), fx=0.5, fy=0.5))
        E = getEssentialMatrix(F, getk())
        poses = ExtractCameraPose(E,getk())

        # plt.plot(0, 0, 'rx')
        # plotCamera(poses[0][1],poses[0][0],'b')
        # plotCamera(poses[1][1],poses[1][0],'g')
        # plotCamera(poses[2][1],poses[2][0],'c')
        # plotCamera(poses[3][1],poses[3][0],'r')


        plt.ylabel('Z')
        plt.xlabel('X')
        total_points = []
        colors = 'bgcr'
        count = []
        for i in range(4):
            points3D = LinearTriangulation(poses[i], inliers, getk())
            plotCamera(poses[i][1],poses[i][0],colors[i])
            plt.plot(points3D[:,0],points3D[:,2],colors[i]+'o')
            total_points.append(points3D)
            count.append(DisambiguateCameraPose(poses[i],points3D))
        bestPose = poses[np.argmax(count)]
        points3D = total_points[np.argmax(count)]
        # visualize(points3D)

        plt.figure()
        plotCamera(bestPose[1],bestPose[0],'r')
        plt.plot(points3D[:,0],points3D[:,2],'ro')
        # cv2.waitKey(0)
       	# plt.show()

        # visualize(points3D, 'linear', file= "output/"+str(key)+".png")
        # print('Non linear triangulation')
        points = NonLinearTraingualtion(bestPose[1], bestPose[0], getk(), inliers[:,0], inliers[:,1], points3D)
        handle =  open('tri.pkl','wb')
        pickle.dump(points,handle)
        handle.close()
    
        pickleFile = open('tri.pkl','rb')
        points = pickle.load(pickleFile)
        points = points[:,:3]
        pickleFile.close()

        R,C = PnpRANSAC(inliers, points,getk())
        plotCamera(R,C,'c')
        plt.plot(points[:,0],points[:,2],'co')
        plt.show()
        # print(np.mean(abs(points3D - points)))
        print('done')

        rotmatrix_to_angles(R)
        # print(key,R,C)

        cv2.waitKey(1)
        # plt.show()

if __name__ == '__main__':
    main()

