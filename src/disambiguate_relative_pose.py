import numpy as np

from src.linear_triangulation import linearTriangulation

def disambiguateRelativePose(Rots,u3,points0,points1,K):
    """ finds the correct relative camera pose (among
     four possible configurations) by returning the one that yields points
     lying in front of the image plane (with positive depth).

     Arguments:
       Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
       u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
       p1   -  Nx2 coordinates of point correspondences in image 1
       p2   -  Nx2 coordinates of point correspondences in image 2
       K   -  3x3 calibration matrix for camera

     Returns:
       R -  3x3 the correct rotation matrix
       T -  3x1 the correct translation vector

       where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
       from the world coordinate system (identical to the coordinate system of camera 1)
       to camera 2.
    """

    # Projection matrix of camera 1
    M1 = K @ np.eye(3,4)

    total_points_in_front_best = 0
    for iRot in range(2):
        R_C2_C1_test = Rots[:,:,iRot]
        
        for iSignT in range(2):
            T_C2_C1_test = u3 * (-1)**iSignT
            
            M2 = K @ np.c_[R_C2_C1_test, T_C2_C1_test]
            p_C1 = linearTriangulation(points0, points1, M1, M2)
            
            # project in both cameras
            p_C2 = np.c_[R_C2_C1_test, T_C2_C1_test] @ p_C1
            
            num_points_in_front1 = np.sum(p_C1[2,:] > 0)
            num_points_in_front2 = np.sum(p_C2[2,:] > 0)
            total_points_in_front = num_points_in_front1 + num_points_in_front2
                  
            if (total_points_in_front > total_points_in_front_best):
                # Keep the rotation that gives the highest number of points
                # in front of both cameras
                R = R_C2_C1_test
                T = T_C2_C1_test
                total_points_in_front_best = total_points_in_front

    return R, T
