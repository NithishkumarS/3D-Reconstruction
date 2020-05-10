import numpy as np
def skew(p):
	v = np.array([[0,-1,p[1]],[1,0,-p[0]],[-p[1],p[0],0]])
	return v

def LinearTriangulation(poses,matches,k):
	points3D = []
	# print(np.shape(matches))
	for i in matches:
		x =  i[0][0]
		y = i[0][1]
		xp = i[1][0]
		yp = i[1][1]
		Xs = []
		for p in poses:
			P1 = np.dot(k,np.hstack((np.eye(3),np.zeros((3,1)))))
			P2 = np.dot(k,np.hstack((p[1],-p[0])))
			skew1 = skew(i[0])
			skew2 = skew(i[1])

			A = np.vstack((np.dot(skew1,P1),np.dot(skew2,P2)))

			# M = np.hstack((p[1],p[0]))
			# Mp = np.hstack((np.eye(3),np.zeros((3,1))))
			# A = [x*M[:,2].T-M[:,0].T,
			# y*M[:,2].T-M[:,1].T,
			# xp*Mp[:,2]-Mp[:,0].T,
			# yp*Mp[:,2]-Mp[:,1].T]
			U,S,Vh = np.linalg.svd(A)
			
			X = Vh[-1,:3]/Vh[-1,-1]
			Xs.append(X)
		points3D.append(Xs)
	return np.array(points3D)
