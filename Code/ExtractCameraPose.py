import numpy as np


def ExtractCameraPose(E):

	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	U,S,V = np.linalg.svd(E)
	C1 = U[:,-1,np.newaxis]
	R1 = np.dot(np.dot(U,W),V)
	C2 = -U[:,-1,np.newaxis]
	R2 = np.dot(np.dot(U,W),V)
	C3 = U[:,-1,np.newaxis]
	R3 = np.dot(np.dot(U,W.T),V.T)
	C4 = -U[:,-1,np.newaxis]
	R4 = np.dot(np.dot(U,W.T),V.T)

	if np.linalg.det(R1) == -1:
		C1 = -1*C1
		R1 = -1*R1

	if np.linalg.det(R2) == -1:
		C2 = -1*C2
		R2 = -1*R2

	if np.linalg.det(R3) == -1:
		C3 = -1*C3
		R3 = -1*R3

	if np.linalg.det(R4) == -1:
		C4 = -1*C4
		R4 = -1*R4


	return [[C1,R1],[C2,R2],[C3,R3],[C4,R4]]