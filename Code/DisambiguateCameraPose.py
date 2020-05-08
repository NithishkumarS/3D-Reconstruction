import numpy as np

def DisambiguateCameraPose(poses,points3D):
	pose_tally = [0,0,0,0]
	# print(np.shape(points3D))
	for X in points3D:
		for num,p in enumerate(poses):
			C = p[0]
			R = p[1]
			if np.dot(R[np.newaxis,2,:],(X[num,np.newaxis].T-C))>0:
				pose_tally[num] +=1
	best_pose = np.argmax(pose_tally)
	return poses[best_pose],points3D[:,best_pose,:]
