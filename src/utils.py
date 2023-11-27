import numpy as np
import cv2
import os


def normalise2DPts(pts):
    """  normalises 2D homogeneous points

     Function translates and normalises a set of 2D homogeneous points
     so that their centroid is at the origin and their mean distance from
     the origin is sqrt(2).

     Usage:   [pts_tilde, T] = normalise2dpts(pts)

     Argument:
       pts -  3xN array of 2D homogeneous coordinates

     Returns:
       pts_tilde -  3xN array of transformed 2D homogeneous coordinates.
       T         -  The 3x3 transformation matrix, pts_tilde = T*pts
    """

    # Convert homogeneous coordinates to Euclidean coordinates (pixels)
    pts_ = pts/pts[2,:]

    # Centroid (Euclidean coordinates)
    mu = np.mean(pts_[:2,:], axis = 1)

    # Average distance or root mean squared distance of centered points
    # It does not matter too much which criterion to use. Both improve the
    # numerical conditioning of the Fundamental matrix estimation problem.
    pts_centered = (pts_[:2,:].T - mu).T

    # Option 1: RMS distance
    sigma = np.sqrt( np.mean( np.sum(pts_centered**2, axis = 0) ) )

    # Option 2: average distance
    # sigma = mean( sqrt(sum(pts_centered.^2)) );

    s = np.sqrt(2) / sigma
    T = np.array([
        [s, 0, -s * mu[0]],
        [0, s, -s * mu[1]],
        [0, 0, 1]])

    pts_tilde = T @ pts_

    return pts_tilde, T


def cross2Matrix(x):
    """ Antisymmetric matrix corresponding to a 3-vector
     Computes the antisymmetric matrix M corresponding to a 3-vector x such
     that M*y = cross(x,y) for all 3-vectors y.

     Input: 
       - x np.ndarray(3,1) : vector

     Output: 
       - M np.ndarray(3,3) : antisymmetric matrix
    """
    M = np.array([[0,   -x[2], x[1]], 
                  [x[2],  0,  -x[0]],
                  [-x[1], x[0],  0]])
    return M



def distPoint2EpipolarLine(F, p1, p2):
    """ Compute the point-to-epipolar-line distance

       Input:
       - F np.ndarray(3,3): Fundamental matrix
       - p1 np.ndarray(3,N): homogeneous coords of the observed points in image 1
       - p2 np.ndarray(3,N): homogeneous coords of the observed points in image 2

       Output:
       - cost: sum of squared distance from points to epipolar lines
               normalized by the number of point coordinates
    """

    N = p1.shape[1]

    homog_points = np.c_[p1, p2]
    epi_lines = np.c_[F.T @ p2, F @ p1]

    denom = epi_lines[0,:]**2 + epi_lines[1,:]**2
    cost = np.sqrt( np.sum( np.sum( epi_lines * homog_points, axis = 0)**2 / denom) / N)

    return cost


def load_init_data(dataset, frames):
    """ Load the initial frames from the dataset

       Input:
       - dataset: the dataset object
       - frames: list of frame indices

       Output:
       - imgs: list of images

    """
    if dataset == 'kitti':
        image1 = cv2.cvtColor(cv2.imread(f'./data/kitti/05/image_0/00000{frames[0]}.png'), cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(cv2.imread(f'./data/kitti/05/image_0/00000{frames[1]}.png'), cv2.COLOR_BGR2GRAY)
        K = np.array(
            [[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02],
            [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02],
            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

    elif dataset == 'parking':
        image1 = cv2.cvtColor(cv2.imread(f'./data/parking/images/img_0000{frames[0]}.png'), cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(cv2.imread(f'./data/parking/images/img_0000{frames[1]}.png'), cv2.COLOR_BGR2GRAY)
        K = np.array(
            [[331.37, 0, 320],
            [0, 369.568, 240],
            [0, 0, 1]])

    elif dataset == 'malaga':
        image1 = cv2.cvtColor(cv2.imread(f'./data/malaga/Images/img_CAMERA1_1261229981.580023_left.jpg'), cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(cv2.imread(f'./data/malaga/Images/img_CAMERA1_1261229981.680019_left.jpg'), cv2.COLOR_BGR2GRAY)
        K = np.array(
            [[1.139837e+03, 0.000000e+00, 6.035000e+02],
            [0.000000e+00, 1.139837e+03, 4.500000e+02],
            [0.000000e+00, 0.000000e+00, 1.000000e+00]])
        K = np.array(
            [[837.619011, 0.000000000000e+00, 522.434637],
             [839.808333, 7.070912000000e+02, 402.367400],
             [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

    return image1, image2, K


def distortPoints(x, D, K):
    """Applies lens distortion D(2) to 2D points x(Nx2) on the image plane. """

    k1, k2 = D[0], D[1]

    u0 = K[0, 2]
    v0 = K[1, 2]

    xp = x[:, 0] - u0
    yp = x[:, 1] - v0

    r2 = xp**2 + yp**2
    xpp = u0 + xp * (1 + k1*r2 + k2*r2**2)
    ypp = v0 + yp * (1 + k1*r2 + k2*r2**2)

    x_d = np.stack([xpp, ypp], axis=-1)

    return x_d


def projectPoints(points_3d, K, D=np.zeros([4, 1])):
    """
    Projects 3d points to the image plane (3xN), given the camera matrix (3x3) and
    distortion coefficients (4x1).
    """
    # get image coordinates
    projected_points = np.matmul(K, points_3d[:, :, None]).squeeze(-1)
    projected_points /= projected_points[:, 2, None]

    # apply distortion
    projected_points = distortPoints(projected_points[:, :2], D, K)

    return projected_points


def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    #
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form
    #           M_tilde = [R_tilde | alpha * t]
    # where R is a rotation matrix. M_tilde encodes the transformation
    # that maps points from the world frame to the camera frame

    # Convert 2D to normalized coordinates
    p_norm = (np.linalg.inv(K) @ np.c_[p, np.ones((p.shape[0], 1))].T).T

    # Build measurement matrix Q
    num_corners = p_norm.shape[0]
    Q = np.zeros((2 * num_corners, 12))

    for i in range(num_corners):
        u = p_norm[i, 0]
        v = p_norm[i, 1]

        Q[2 * i, 0:3] = P[i, :]
        Q[2 * i, 3] = 1
        Q[2 * i, 8:11] = -u * P[i, :]
        Q[2 * i, 11] = -u

        Q[2 * i + 1, 4:7] = P[i, :]
        Q[2 * i + 1, 7] = 1
        Q[2 * i + 1, 8:11] = -v * P[i, :]
        Q[2 * i + 1, 11] = -v

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    u, s, v = np.linalg.svd(Q, full_matrices=True)
    M_tilde = np.reshape(v.T[:, -1], (3, 4));

    # Extract [R | t] with the correct scale
    if (np.linalg.det(M_tilde[:, :3]) < 0):
        M_tilde *= -1

    R = M_tilde[:, :3]

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    u, s, v = np.linalg.svd(R)
    R_tilde = u @ v

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    alpha = np.linalg.norm(R_tilde, 'fro') / np.linalg.norm(R, 'fro')

    # Build M_tilde with the corrected rotation and scale
    M_tilde = np.c_[R_tilde, alpha * M_tilde[:, 3]]

    return M_tilde


def load_image_paths(dataset, num_frames):
    """Loads the image paths for the given dataset and number of frames.
    """
    if dataset == 'kitti':
        image_paths = [f"./data/kitti/05/image_0/{i}" for i in sorted(os.listdir('./data/kitti/05/image_0'))]
    elif dataset == 'parking':
        image_paths = [f"./data/parking/images/{i}" for i in sorted(os.listdir('./data/parking/images'))]
    elif dataset == 'malaga':
        image_paths = [f"./data/malaga/Images/{i}" for i in sorted(os.listdir('./data/malaga/Images')) if i.endswith('left.jpg')]
    else:
        raise ValueError('Unknown dataset')

    if num_frames == 'all':
        return image_paths
    else:
        return image_paths[:num_frames+2]

