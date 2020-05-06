
def DisambiguateCameraPose(poses,points3D):
	pose_tally = [0,0,0,0]
	for X in points3D:
		for num,p in enumerate(poses):
			C = p[0]
			R = p[1]
			if R[2,:]@(X[num]-C)>0:
				pose_tally[num] +=1
	best_pose = np.argmax(pose_tally)
	return poses[best_pose]