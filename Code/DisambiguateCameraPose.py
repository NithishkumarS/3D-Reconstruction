import numpy as np

def DisambiguateCameraPose(poses,points3D):
	# pose_tally = [0,0,0,0]
	# # print(np.shape(points3D))
	# for X in points3D:
	# 	for num,p in enumerate(poses):
	# 		C = p[0]
	# 		R = p[1]
	# 		if np.dot(R[np.newaxis,1,:],(X[num,np.newaxis].T-C))>0:
	# 			pose_tally[num] +=1
	# best_pose = np.argmax(pose_tally)
	# return poses[best_pose],points3D[:,best_pose,:]
	# def findZ(R1,t,pts1,pts2,K):
	Co = [[0],[0],[0]]
	Ro = np.eye(3,3)
	R1 = poses[1]
	C1 = poses[0]
	P1 = np.eye(3,4)
	P2 = np.hstack((R1,C1))
	X1 = points3D
	X1 = np.array(X1)
	
	check = 0
	for i in range(X1.shape[0]):
		x = X1[i,:].reshape(-1,1)
		if np.dot(R1[2],np.subtract(x[0:3],C1)) > 0 and x[2] > 0:
			check += 1

	return check