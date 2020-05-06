import numpy as np

def computeFundamentalMatrix(_X1, _X2):
    # print(np.shape(_X1))
    # print(_X1)
    A = np.empty((0,9))
    for x1,x2 in zip(_X1, _X2):
        ele = np.array([[x1[0]*x2[0], x1[0]*x2[1], x1[0], x1[1]*x2[0], x1[1]*x2[1], x1[1], x2[0], x2[1], 1]])
        A = np.vstack((A,ele))
    u, s, vh = np.linalg.svd(A)
    F = vh[-1, :] / vh[-1, -1]
    F = F.reshape((3,3))
    # print('b:::', F)
    return F

def test_func():
    X1 = [[1,1],[3,3]]
    X2 = [[2,2], [4,4]]
    res = computeFundamentalMatrix(X1, X2)
    # if res == [-0.1977383 , -0.05495667, -0.02747834, -0.05495667, -0.05495667, -0.02747834,  -0.05495667, -0.05495667, 1.]:
    #     print('test passes')

# test_func()

    