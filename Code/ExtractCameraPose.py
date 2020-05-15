import numpy as np


def ExtractCameraPose(E,K):

	# W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	# U,S,V = np.linalg.svd(E)
	# C1 = U[:,-1,np.newaxis]
	# R1 = np.dot(np.dot(U,W),V)
	# C2 = -U[:,-1,np.newaxis]
	# R2 = np.dot(np.dot(U,W),V)
	# C3 = U[:,-1,np.newaxis]
	# R3 = np.dot(np.dot(U,W.T),V)
	# C4 = -U[:,-1,np.newaxis]
	# R4 = np.dot(np.dot(U,W.T),V)

	# if np.linalg.det(R1) == -1:
	# 	C1 = -1*C1
	# 	R1 = -1*R1

	# if np.linalg.det(R2) == -1:
	# 	C2 = -1*C2
	# 	R2 = -1*R2

	# if np.linalg.det(R3) == -1:
	# 	C3 = -1*C3
	# 	R3 = -1*R3

	# if np.linalg.det(R4) == -1:
	# 	C4 = -1*C4
	# 	R4 = -1*R4


	# return [[C1,R1],[C2,R2],[C3,R3],[C4,R4]]


	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	U,S,V = np.linalg.svd(E)

	poses = {}

	poses['C1'] = U[:,2].reshape(3,1)
	poses['C2'] = -U[:,2].reshape(3,1)
	poses['C3'] = U[:,2].reshape(3,1)
	poses['C4'] = -U[:,2].reshape(3,1)

	poses['R1'] = np.dot(np.dot(U,W),V)
	poses['R2'] = np.dot(np.dot(U,W),V) 
	poses['R3'] = np.dot(np.dot(U,W.T),V)
	poses['R4'] = np.dot(np.dot(U,W.T),V)

	for i in range(4):
		C = poses['C'+str(i+1)]
		R = poses['R'+str(i+1)]
		if np.linalg.det(R) < 0:
			C = -C 
			R = -R 
			poses['C'+str(i+1)] = C 
			poses['R'+str(i+1)] = R
		I = np.eye(3,3)
		M = np.hstack((I,C.reshape(3,1)))
		poses['P'+str(i+1)] = np.dot(np.dot(K,R),M)

	return [[poses['C1'],poses['R1']],[poses['C2'],poses['R2']],[poses['C3'],poses['R3']],[poses['C4'],poses['R4']]]