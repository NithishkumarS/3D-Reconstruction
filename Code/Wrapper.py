#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project: Structure from Motion

Author(s):
Nithish Kumar

Zack

Needed Directories

YourDirectoryID_hw1.zip
│   README.md
|   Your Code files
|   ├── GetInliersRANSAC.py
|   ├── EstimateFundamentalMatrix.py
|   ├── EssentialMatrixFromFundamentalMatrix.py
|   ├── ExtractCameraPose.py
|   ├── LinearTriangulation.py
|   ├── DisambiguateCameraPose.py
|   ├── NonlinearTriangulation.py
|   ├── PnPRANSAC.py
|   ├── NonlinearPnP.py
|   ├── BuildVisibilityMatrix.py
|   ├── BundleAdjustment.py
|   ├── Wrapper.py
|   ├── Any subfolders you want along with files
|   Wrapper.py
|   Data
|   ├── BundleAdjustmentOutputForAllImage
|   ├── FeatureCorrespondenceOutputForAllImageSet
|   ├── LinearTriangulationOutputForAllImageSet
|   ├── NonLinearTriangulationOutputForAllImageSet
|   ├── PnPOutputForAllImageSetShowingCameraPoses
|   ├── Imgs/
└── Report.pdf

"""

# Code starts here:
import argparse
import numpy as np
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import utils.utils as ut
from features import *
import copy
import itertools
import random
from EssentialMatrixFromFundamentalMatrix import getEssentialMatirx
from ExtractCameraPose import *
from LinearTriangulation import *
def getk():
    K = [[568.996140852, 0, 643.21055941],
    [0, 568.988362396, 477.982801038],
    [0, 0, 1]] 
    return K

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--ImgDirectory', default='../Data/Train/Set1/',
                        help='Directory that contains images for panorama sticking')

    Parser.add_argument('--compute_corners', default=True, help='Directory that contains images for panorama sticking')
    Args = Parser.parse_args()
    ImgDirectory = Args.ImgDirectory
    compute_corners = Args.compute_corners



    E = getEssentialMatrix(F,getk())
    poses = ExtractCameraPose(E)
    points3D= LinearTriangulation(poses,matches)
    bestPose = DisambiguateCameraPose(poses,points3D)


if __name__ == '__main__':
    main()

