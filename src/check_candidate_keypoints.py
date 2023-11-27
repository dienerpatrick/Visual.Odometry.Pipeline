import numpy as np
import cv2

def check_candidate_keypoints(C, F, T, pose, K, threshold):
    """
    Check candidate keypoints for triangulation
    :param C: [np.ndarray] [Nx2] candidate keypoints
    :param F: [np.ndarray] [Nx2] candidate keypoints first occurance
    :param T: [np.ndarray] [Nx12] first occurance projection matrix
    :param pose: [np.ndarray] [1x12] current projection matrix
    :param K: [np.ndarray] [3x3] camera calibration matrix
    :param threshold: [float] angle threshold for triangulation in radians
    :return: [np.ndarray]
    """
    # print(T.shape)
    status = np.zeros(C.shape[0])

    for i in range(C.shape[0]):
        # calculate projection matrix
        P1 = np.dot(K, T[i].reshape(3, 4))
        P2 = np.dot(K, pose.reshape(3, 4))

        # keypoint pixel coordinates to camera coordinates
        x1 = np.dot(np.linalg.inv(K), np.array([F[i, 0], F[i, 1], 1]))
        x2 = np.dot(np.linalg.inv(K), np.array([C[i, 0], C[i, 1], 1]))

        # camera coordinates to world coordinates
        X1 = np.dot(np.linalg.pinv(P1), np.array([x1[0], x1[1], x1[2]]))[:3]
        X2 = np.dot(np.linalg.pinv(P2), np.array([x2[0], x2[1], x2[2]]))[:3]

        # calculate angle between two vectors
        angle = np.arccos(np.dot(X1, X2) / (np.linalg.norm(X1) * np.linalg.norm(X2)))

        # check if angle is bigger than threshold
        if angle > threshold:
            status[i] = 1

    return status















