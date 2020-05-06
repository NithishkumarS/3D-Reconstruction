import numpy as np


def LinearTriangulation(poses,matches):
	points3D = []
	for i in matches:
		x =  matches[0,0]
		y = matches[0,1]
		xp = matches[1,0]
		yp = matches[1,1]
		for p in poses:
			Xs = []
			M = np.hstack((p[1],p[0]))
			Mp = np.hstack((np.eye(3),np.zeros((3,1))))
			A = [x*M[:,2].T-M[:,0].T,
			y*M[:,2].T-M[:,1].T,
			xp*Mp[:,2]-Mp[:,0].T,
			yp*Mp[:,2]-Mp[:,1].T]
			U,S,V = np.linalg.svd(A)
			X = V.T[:,-1]
			Xs.append(X)
		points3D.append(Xs)
	return np.array(points3D).T
