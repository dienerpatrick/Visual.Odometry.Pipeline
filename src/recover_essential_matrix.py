if __name__ == "__main__":
    from utils import normalise2DPts
else:
    from src.utils import normalise2DPts

import numpy as np
import cv2


def recover_essential_matrix(points1, points2, K):
    """
    Recover the essential matrix from the given points
    :param p1:
    :param p2:
    :return:
    """

    # normalize each set of points so that the origin is at centroid and mean distance from origin is sqrt(2)
    p1, T1 = normalise2DPts(points1)
    p2, T2 = normalise2DPts(points2)

    # Sanity checks
    assert (p1.shape == p2.shape), "Input points dimension mismatch"
    assert (p1.shape[0] == 3), "Points must have three columns"

    num_points = p1.shape[1]
    assert (num_points >= 8), \
        'Insufficient number of points to compute fundamental matrix (need >=8)'

    # Compute the measurement matrix A of the linear homogeneous system whose
    # solution is the vector representing the fundamental matrix.
    A = np.zeros((num_points, 9))
    for i in range(num_points):
        A[i, :] = np.kron(p1[:, i], p2[:, i]).T

    # "Solve" the linear homogeneous system of equations A*f = 0.
    # The correspondences x1,x2 are exact <=> rank(A)=8 -> there exist an exact solution
    # If measurements are noisy, then rank(A)=9 => there is no exact solution,
    # seek a least-squares solution.
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    F = np.reshape(vh[-1, :], (3, 3)).T

    # Enforce det(F)=0 by projecting F onto the set of 3x3 singular matrices
    u, s, vh = np.linalg.svd(F)
    s[2] = 0
    F = u @ np.diag(s) @ vh

    # Undo the normalization
    F = T2.T @ F @ T1

    E = K.T @ F @ K

    return E


def recover_essential_matrix2(points1, points2, K):

    E, status = cv2.findEssentialMat(points1, points2, K,  cv2.FM_RANSAC, 0.95)

    inliers1 = []
    inliers2 = []

    for i, qual in enumerate(status):
        if qual:
            inliers1.append(points1[i])
            inliers2.append(points2[i])

    inliers1 = np.squeeze(np.float32(inliers1))
    inliers2 = np.squeeze(np.float32(inliers2))


    num_inliers = np.sum(status)
    total = len(status)

    print("Number of inliers: ", num_inliers, "out of ", total)
    print("Essential matrix: ", E)

    return E, inliers1, inliers2




