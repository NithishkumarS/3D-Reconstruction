import numpy as np

def getEssentialMatrix(F,K):
	E = np.matmul(np.matmul(K.T,F),K)
	U,S,V = np.linalg.svd(E)
	singulars = [[S[0],0,0],[0,S[1],0],[0,0,0]]
	# singulars = [[1,0,0],[0,1,0],[0,0,0]]

	E = np.matmul(np.matmul(U,singulars),V)
	return E