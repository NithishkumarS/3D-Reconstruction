import numpy as np
import random
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import copy 

# print(cv2.__version__)
import fileinput, optparse
# from matplotlib import pyplot as plt
from EstimateFundamentalMatrix import *

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
    
def drawMatches(matches,combed_img, width,color):
    matches = np.array(matches)
    print(np.shape(matches))
    matches[:,1,0] = matches[:,1,0]+width-1
    for match in matches:
        y = int(match[0,1])
        x = int(match[0,0])
        yp = int(match[1,1])
        xp = int(match[1,0])
        combed_img = cv2.line(combed_img,(x,y),(xp,yp),color)
    return combed_img


def getInliersRANSAC(M,images):
    Matches = importMatches()
    Data = {}
    episilon = .2
    for key,matches  in Matches.items():
        # print(key)
        # if key != '12':
        #     continue
        
        image1 = images[int(key[0])-1]
        image2 = images[int(key[1])-1]
        width = np.shape(image1)[1]
        combed_img = np.hstack((image1,image2))
        orig_matches = copy.copy(np.array(matches))

        drawnMatches = drawMatches(matches,combed_img, width,[0,0,255])

        c1 = np.hstack((np.array(matches)[:,0], np.ones((len(matches),1))) )
        c2 = np.hstack((np.array(matches)[:,1], np.ones((len(matches),1))) )
        S_inliers = []
        S_points_inliers = []
        n = 0
        best_F =[]
        for i in range(M):
            l = range(len(c2))
            rand_idx = random.sample(l, k=8)
            # print(type(c1[rand_idx]))
            # F = computeFundamentalMatrix(c1[rand_idx], c2[rand_idx])
            F = fundamental_matrix(c1[rand_idx],c2[rand_idx])
            S = []
            S_points = []
            for j in range(len(c1)):
                x1, x2  = c1[j],c2[j]

                ep1 = np.dot(F,x1)
                ep2 = np.dot(F.T,x2)
                numerator = np.dot(np.dot(x2.T, F),x1)
                denominator = ep1[0]**2 + ep1[1]**2 + ep2[0]**2 + ep2[1]**2
                e = numerator**2 / denominator
                if e <= episilon:
                    # print(abs(np.dot(np.dot(x2.T, F),x1)))
                    S.append(j)
                    S_points.append([x1[:2],x2[:2]])
                    # cv2.waitKey(0)
                
            if n <len(S):
                n = len(S)
                S_inliers = S
                S_points_inliers = S_points
                best_F = F
            
            if float(n)/len(matches) >0.8:
                print('break')
                break

        print('done')
        X1 = []
        X2 = []
        l = range(len(S))
        if len(S) >=8:
            for r in range(len(S)):
                X1.append(c1[S_inliers[r]])
                X2.append(c2[S_inliers[r]])
            # F= computeFundamentalMatrix(np.array(X1), np.array(X2))
            # print(np.shape(X1))
            # print(np.shape(X2))
            F = fundamental_matrix(np.array(X1),np.array(X2))
            drawn_inliers = drawMatches(S_points_inliers,drawnMatches, width,[0,255,0])
        else:
            F = best_F
            drawn_inliers = drawnMatches

        
        # cv2.imshow('inliers',cv2.resize(drawn_inliers,(0,0), fx=0.5,fy=0.5))
        # print(len(matches),len(S_points_inliers))
        # cv2.waitKey(0)
        cv2.imwrite("matches"+key+'.jpg',cv2.resize(drawn_inliers,(0,0), fx=0.5,fy=0.5))
        print(key)
        print(100.0*len(S_points_inliers)/len(matches),'%')
        Data[key] = [F,S_points_inliers,drawn_inliers,orig_matches]
    return Data

# b = getInliersRANSAC(cv2.imread('../Data/1.jpg'),cv2.imread('../Data/2.jpg') , 1)
