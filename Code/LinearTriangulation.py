import numpy as np


def LinearTriangulation(poses,matches):
	points3D = []
	# print(np.shape(matches))
	for i in matches:
		x =  i[0][0]
		y = i[0][1]
		xp = i[1][0]
		yp = i[1][1]
		Xs = []
		for p in poses:
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
	return np.array(points3D)
