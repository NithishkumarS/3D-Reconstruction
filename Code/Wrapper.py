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
from NonlinearPnp import *
import matplotlib.pyplot as plt
import pickle
from PySBA import *

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

def camera2vector(R,C,K):
    V = np.zeros((9))
    t = cv2.Rodrigues(R)[0].T[0]
    V[:3] = t
    V[3:6] = C[:,0]
    V[6] = K[0,0]
    V[7] = K[0,2]
    V[8] = K[1,2]

    return V

def appendList(vec,ele):
    vec = list(vec)
    vec.append(ele)
    return vec

def reprojection_error(xs, Xs, P):
    # X = np.concatenate([X,np.array([1])])
    
    print(np.shape(xs[0]))
    print(np.shape(Xs[0]))
    error = 0
    for (x,X) in zip(xs,Xs):
        if len(Xs[0]) == 3:
            X = np.concatenate([X, np.array([1])])
        error += (x[0]- np.dot(P[0],X) / np.dot(P[2], X))**2 + (x[1]- np.dot(P[1],X) / np.dot(P[2], X))**2
    return  error/len(xs)

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
    key = "12"
    info = data[key]
    data.pop(key,None)
    
    F = info[0]
    inliers = np.array(info[1])
    matches = info[2]
    cv2.imshow('matches', cv2.resize(matches, (0, 0), fx=0.5, fy=0.5))
    E = getEssentialMatrix(F, getk())
    poses = ExtractCameraPose(E,getk())

    plt.ylabel('Z')
    plt.xlabel('X')
    total_points = []
    colors = 'bgcrmy'
    count = []
    for i in range(4):
        points3D = LinearTriangulation(poses[i][1],poses[i][0], inliers, getk())
        plotCamera(poses[i][1],poses[i][0],colors[i])
        # plt.plot(points3D[:,0],points3D[:,2],colors[i]+'o')
        total_points.append(points3D)
        count.append(DisambiguateCameraPose(poses[i],points3D))
    bestPose = poses[np.argmax(count)]
    points3D = total_points[np.argmax(count)]
    P = np.dot(getk(),np.hstack((bestPose[1], -np.dot(bestPose[1],bestPose[0]))))
    error1 = reprojection_error(inliers[:,0],points3D,P)
    error2 = reprojection_error(inliers[:,1],points3D,P)
    error = .5*(error1+error2)
    print('Linear reprojection',error)
    # visualize(points3D)
    # plt.show()
    plt.figure()
    # plotCamera(bestPose[1],bestPose[0],'r')
    plt.plot(points3D[:,0],points3D[:,2],'go')
    plt.savefig('../Output/linear'+key+'.png')
    # cv2.waitKey(0)
  	# plt.show()

    print('Non linear triangulation')
    points = NonLinearTraingualtion(bestPose[1], bestPose[0], getk(), inliers[:,0], inliers[:,1], points3D)
    handle =  open('tri.pkl','wb')
    pickle.dump(points,handle)
    handle.close()

    pickleFile = open('tri.pkl','rb')
    points = pickle.load(pickleFile)
    pickleFile.close()
    plt.plot(points[:,0],points[:,2],'ro')
    plt.savefig('../Output/nonlinear'+key+'.png')

    points3D = points[:,:3]

    error1 = reprojection_error(inliers[:,0],points,P)
    error2 = reprojection_error(inliers[:,1],points,P)
    error = .5*(error1+error2)
    print('Non-Linear reprojection',error)

    points = points[:,:3]

    # Bundle Adjustment Setup
    cameraArray = []
    cameraArray.append(camera2vector(np.eye(3),np.zeros((3,1)),getk()))
    cameraArray.append(camera2vector(bestPose[1],bestPose[0],getk()))


    # observations = points
    points_ind = range(len(points))
    points_ind.extend(range(len(points)))
    camera_ind = np.zeros((2*len(points3D)),dtype=int)
    camera_ind[len(points3D):] += 1
    camera_ind = list(camera_ind)
    points2D = list(np.vstack((inliers[:,0],inliers[:,1])))

    Rset = []
    Cset = []
    Rset.append(bestPose[1])
    Cset.append(bestPose[0])
    inc = 0
    params =[0,0,0]

    for key, info in data.items():
        cameraPoints = []
        worldPoints = []
        print(key)
        if key[0] != '1' and key[0] != '2' :
            print('Broke')
            continue
        initial_pose = cameraArray[int(key[0])-1]
        R2 = cv2.Rodrigues(initial_pose[:3])[0]
        C2 = np.array(initial_pose[3:6,np.newaxis])
        matches = np.array(info[3])
        # print(matches[:15])
        ele = int(key[0])-1
        i_inc = 0
        points3D_inc = len(points3D)
        for i in inliers:
            for j in matches:
                if list(i[ele]) == list(j[0]):
                    cameraPoints.append([j[0],j[1]])
                    worldPoints.append(points[i_inc])
                    camera_ind.append(len(cameraArray))
                    points_ind.append(i_inc)
                    points2D.append([j[1][0],j[1][1]])
                    # Adds camera index
                    # camera_ind[i_inc] = appendList(camera_ind[i_inc],int(key[1])-1)
                    # points_ind[i_inc] = appendList(points_ind[i_inc],points3D_inc)
                    
                    # points2D[i_inc][0] = appendList(points2D[i_inc][0],j[1][0])
                    # points2D[i_inc][1] = appendList(points2D[i_inc][0],j[1][1])
                    points3D_inc += 1
                    break
            i_inc += 1

        # Estimate New Camera Position
        R,C = PnpRANSAC(cameraPoints, worldPoints,getk())
        P = np.dot(getk(),np.hstack((R, -np.dot(R,C[:,np.newaxis]))))
        error = reprojection_error(np.array(cameraPoints)[:,1],worldPoints,P)
        print('Linear PnP',error)

        R,C = NonLinearPnp(R, C, getk(), np.array(cameraPoints)[:,1], worldPoints)
        P = np.dot(getk(),np.hstack((R, -np.dot(R,C))))
        error = reprojection_error(np.array(cameraPoints)[:,1],worldPoints,P)
        print('Non-Linear PnP',error)
        cameraArray.append(camera2vector(R,C,getk()))
        # Project New 3D Points
        worldPoints = LinearTriangulation(R,C, cameraPoints, getk(),R2,C2)
        worldPoints = NonLinearTraingualtion(R, C, getk(), inliers[:,0], inliers[:,1], points3D,R2,C2)
        error = reprojection_error(np.array(cameraPoints)[:,1],worldPoints,P)
        print('Before BA',error)
        
        # Plot Camera and Points for latest Frame
        # plt.plot(worldPoints[:,0],worldPoints[:,2],colors[inc]+'o')
        plt.figure
        plotCamera(R,C,colors[inc])
        plt.savefig('../Output/pnp'+key+'.png')

        # Bundle Adjustment Setup
        points3D = np.vstack((points3D,worldPoints))

        pysba = PySBA(np.array(cameraArray),points3D,points2D, np.array(camera_ind),np.array(points_ind))
        params = pysba.bundleAdjust()
        error = reprojection_error(np.array(cameraPoints),params[1][-len(cameraPoints):],P)
        print('Non-Linear PnP',error)
        plt.figure()
        plt.plot(params[1][0],params[1][2],'ko')
        plt.savefig('../Output/BA'+key+'.png')
        # plt.show()
        inc += 1
    
    points3D = params[1]
    plt.plot(points3D[0],points3D[2],'ko')

        # print(np.mean(abs(points3D - points)))
    # print('done')

    # rotmatrix_to_angles(R)
        # print(key,R,C)

    cv2.waitKey(1)
        # plt.show()

if __name__ == '__main__':
    main()

