import numpy as np


def ExtractCameraPose(E,K):

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