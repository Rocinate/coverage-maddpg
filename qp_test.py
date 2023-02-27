import numpy as np
from qpsolvers import solve_qp
H = np.array([[1., -1.], [-1., 2.]])
f = np.array([[-2.], [-6.]]).reshape((2,))
L = np.array([[1., 1.], [-1., 2.], [2., 1.]])
k = np.array([[2.], [2.], [3.]]).reshape((3,))

x = solve_qp(H, f, L, k, solver="cvxopt")
print("QP solution: x = {}".format(x))
