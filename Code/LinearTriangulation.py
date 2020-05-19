import numpy as np
def skew(p):
	v = np.array([[0,-1,p[1]],[1,0,-p[0]],[-p[1],p[0],0]])
	return v

def LinearTriangulation(R2,C2,matches,K,R1=np.eye(3),C1=np.zeros((3,1))):

	P1 = np.dot(K,np.hstack((R1, -np.dot(R1,C1))))
	P2 = np.dot(K,np.hstack((R2, -np.dot(R2,C2))))
	
	X = []

	for pair in matches:
		x1 = pair[0]
		x2 = pair[1]

		a1 = x1[0]*P1[2,:]-P1[0,:]
		a2 = x1[1]*P1[2,:]-P1[1,:]
		a3 = x2[0]*P2[2,:]-P2[0,:]
		a4 = x2[1]*P2[2,:]-P2[1,:]
		
		A = [a1,a2,a3,a4]		

		# print(A)

		# print(np.shape(A))

		U,S,V = np.linalg.svd(A)
		V_out = V[3]
		# V_out = V_out.reshape(-1,1)
		V_out = V_out/V_out[-1]
		X.append(V_out)

	return np.array(X)