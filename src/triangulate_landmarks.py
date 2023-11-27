import numpy as np
import cv2
import matplotlib.pyplot as plt


from src.recover_essential_matrix import recover_essential_matrix2
from src.decompose_essential_matrix import decomposeEssentialMatrix
from src.disambiguate_relative_pose import disambiguateRelativePose
from src.linear_triangulation import linearTriangulation


def triangulate_landmarks(features0, features1, K):
    
    E, inliers1, inliers2 = recover_essential_matrix2(features0, features1, K)

    # Extract the relative camera positions (R,T) from the essential matrix
    Rots, u3 = decomposeEssentialMatrix(E)

    # Disambiguate among the four possible configurations
    R, T = disambiguateRelativePose(Rots, u3, inliers1, inliers2, K)

    # project points to 3d point cloud
    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R, T]

    P = linearTriangulation(inliers1, inliers2, M1, M2)
    P_mask_positive_Z = P[2,:]>0
    P = P[:,P_mask_positive_Z==1]

    inliers1 = inliers1[P_mask_positive_Z==1,:]
    inliers2 = inliers2[P_mask_positive_Z==1,:]

    return P, inliers1, inliers2, R, T


                       
