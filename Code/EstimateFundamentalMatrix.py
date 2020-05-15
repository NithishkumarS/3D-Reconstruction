import numpy as np
import cv2

def computeFundamentalMatrix(_X1, _X2):
    # print(np.shape(_X1))
    # print('in',type(_X1))
    A = np.empty((0,9))
    for x1,x2 in zip(_X1, _X2):
        u1 = x1[0]
        u2 = x2[0]
        v1 = x1[1]
        v2 = x2[1]
        # ele = np.array([u2*u1,u2*v1,u2,v2*u1,v2*v1,v2,u1,v1,1])

        ele = np.array([[x1[0]*x2[0], x1[0]*x2[1], x1[0], x1[1]*x2[0], x1[1]*x2[1], x1[1], x2[0], x2[1], 1]])
        A = np.vstack((A,ele))

    u, s, vh = np.linalg.svd(A)
    F = vh[-1, :] / vh[-1, -1]
    F = F.reshape((3,3))
    
    U,S,Vh = np.linalg.svd(F)
    singulars = [[S[0],0,0],[0,S[1],0],[0,0,0]]
    # singulars = [[1,0,0],[0,1,0],[0,0,0]]
    F = np.matmul(np.matmul(U,singulars),Vh)

    # print(type(_X1))
    F_test,_ = cv2.findFundamentalMat(_X1,_X2,cv2.FM_8POINT)
    
    # print(np.dot(np.dot(x1.T,F),x2))

    # print(np.dot(np.dot(x1.T,F_test),x2))
    # print(np.linalg.matrix_rank(F))
    # print(np.linalg.matrix_rank(F_test))
    # print(F)
    # print(F_test)
    # print(zach)
    return F_test


def fundamental_matrix(points1,points2):
    # F_b,_ = cv2.findFundamentalMat(points1,points2,cv2.FM_8POINT)

    mat = []
    mass_cent = [0.,0.]
    mass_cent_p = [0.,0.]
    for i in range(len(points1)):
        mass_cent[0] += points1[i,0]
        mass_cent[1] += points1[i,1]
        mass_cent_p[0] += points2[i,0]
        mass_cent_p[1] += points2[i,1]
    mass_cent = np.divide(mass_cent,float(len(points1)))
    mass_cent_p = np.divide(mass_cent_p,float(len(points1)))

    scale1 = 0.
    scale2 = 0.
    for i in range(len(points1)):
        scale1 += np.sqrt((points1[i][0]-mass_cent[0])**2+(points1[i][1]-mass_cent[1])**2)
        scale2 += np.sqrt((points2[i][0]-mass_cent_p[0])**2+(points2[i][1]-mass_cent_p[1])**2)
    
    scale1 = scale1/len(points1)
    scale2 = scale2/len(points1)

    scale1 = np.sqrt(2.)/scale1
    scale2 = np.sqrt(2.)/scale2
    A = np.zeros((8,9))
    for i in range(8):
        x1 = (points1[i][0]-mass_cent[0])*scale1
        y1 = (points1[i][1]-mass_cent[1])*scale1
        x2 = (points2[i][0]-mass_cent_p[0])*scale2
        y2 = (points2[i][1]-mass_cent_p[1])*scale2

        row = np.array([x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,1])
        A[i]=row

    U,S,V = np.linalg.svd(A)

    F = V[-1]
    F = np.reshape(F,(3,3))
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(np.dot(U,np.diag(S)),V)

    T1 = np.array([scale1,0,-scale1*mass_cent[0],0,scale1,-scale1*mass_cent[1],0,0,1])
    T1 = T1.reshape((3,3))
    T2 = np.array([scale2,0,-scale2*mass_cent_p[0],0,scale2,-scale2*mass_cent_p[1],0,0,1])
    T2 = T2.reshape((3,3))
    F = np.dot(np.dot(np.transpose(T2),F),T1)
    F = F / F[-1,-1]
    return F

def test_func(_X1, _X2, c1,c2):
    # X1 = [[1,1],[3,3]]
    # X2 = [[2,2], [4,4]]
    # res = computeFundamentalMatrix(X1, X2)
    # pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(c1, c2, cv2.FM_LMEDS)  #np.array(_X1), np.array(_X2)
    F_computed = computeFundamentalMatrix(_X1, _X2)
    print('function: ')
    print(F)
    print('Computed: ')
    print(F_computed)

    # We select only inlier points
    # pts1 = pts1[mask.ravel() == 1]
    # pts2 = pts2[mask.ravel() == 1]
    # if res == [-0.1977383 , -0.05495667, -0.02747834, -0.05495667, -0.05495667, -0.02747834,  -0.05495667, -0.05495667, 1.]:
    #     print('test passes')

# test_func()

    