import numpy as np

from src.utils import cross2Matrix

def linearTriangulation(p1, p2, M1, M2):
    """ Linear Triangulation
     Input:
      - p1 np.ndarray(N, 3): coordinates of points in image 1
      - p2 np.ndarray(N, 3): coordinates of points in image 2
      - M1 np.ndarray(3, 4): projection matrix corresponding to first image
      - M2 np.ndarray(3, 4): projection matrix corresponding to second image

     Output:
      - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """

    # convert featuures to homogenous coordinates and flip to (3, N)
    p1_h = np.c_[p1, np.ones((p1.shape[0]))].transpose()
    p2_h = np.c_[p2, np.ones((p2.shape[0]))].transpose()

    assert(p1_h.shape == p2_h.shape), "Input points dimension mismatch"
    assert(p1_h.shape[0] == 3), "Points must have three columns"
    assert(M1.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"
    assert(M2.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"

    num_points = p1_h.shape[1]
    P = np.zeros((4, num_points))

    # Linear Algorithm
    for i in range(num_points):
        # Build matrix of linear homogeneous system of equations
        A1 = cross2Matrix(p1_h[:, i]) @ M1
        A2 = cross2Matrix(p2_h[:, i]) @ M2
        A = np.r_[A1, A2]

        # Solve the homogeneous system of equations
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        P[:, i] = vh.T[:,-1]

    # Dehomogenize (P is expressed in homoegeneous coordinates)
    P /= P[3,:]

    return P
    


