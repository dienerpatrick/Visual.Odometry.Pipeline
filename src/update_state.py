import matplotlib
from src.extract_keypoints import extract_keypoints
from src.track_keypoints import track_keypoints
from src.extract_candidate_keypoints import extract_candidate_keypoints
from src.triangulate_landmarks import triangulate_landmarks
from src.ransac_localization import ransacLocalization
from src.check_candidate_keypoints import check_candidate_keypoints
from src.linear_triangulation import linearTriangulation
from src.utils import load_init_data, load_image_paths
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


def update_state(S, old_frame, new_frame, K, angle_threshold):
    """
    Update state S with new frame
    :param S: [dict] state
    :param old_frame: [np.ndarray] old frame
    :param new_frame: [np.ndarray] new frame
    :return: updated state
    """

    # Track features from previous frame to current frame and update state
    tracked_P, status = track_keypoints(S['P'], old_frame, new_frame)

    S['P'] = tracked_P[status == 1]
    S['X'] = S['X'][status == 1]
    print("\n---TRACKING KEYPOINTS---")
    print("Number of tracked keypoints: {}".format(len(tracked_P)))
    print("Number of successfully tracked keypoints: {}".format(sum(status)))

    # Track candidates from previous frame to current frame and update state
    if S['C'].shape[0] > 0:
        tracked_C, status = track_keypoints(S['C'], old_frame, new_frame)

        S['C'] = tracked_C[status == 1]
        S['F'] = S['F'][status == 1]
        S['T'] = S['T'][status == 1]

        print("\n---TRACKING CANDIDATES---")
        print("Number of tracked candidate keypoints: {}".format(len(tracked_C)))
        print("Number of successfully tracked candidate keypoints: {}".format(sum(status)))

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(S['X'], S['P'], K, distCoeffs=None, useExtrinsicGuess=True,
                                                     iterationsCount=10000, reprojectionError=5.0, confidence=0.99,
                                                     flags=cv2.SOLVEPNP_EPNP)
    status_P = np.zeros((len(S['P'])))

    for i in inliers:
        status_P[i] = 1
    S['P'] = S['P'][status_P == 1]
    S['X'] = S['X'][status_P == 1]
    rmat, jacobian = cv2.Rodrigues(rvec)

    T_current = np.c_[rmat, tvec].flatten()

    # Check candidate keypoints for triangulation
    if S['C'].shape[0] > 0:
        c_status = check_candidate_keypoints(S['C'], S['F'], S['T'], T_current, K, angle_threshold)
        print("\n---CHECKING CANDIDATES---")
        print("Number of checked candidate keypoints: {}".format(len(S['C'])))
        print("Number of candidate keypoints valid for triangulation: {}".format(sum(c_status)))

        for i in range(len(c_status)):
            if c_status[i] == 1:
                # Triangulate new landmark and add it to the state
                X_new = linearTriangulation(np.expand_dims(S['F'][i], axis=0), np.expand_dims(S['C'][i], axis=0),
                                            K @ T_current.reshape(3, 4), K @ S['T'][i].reshape(3, 4))

                if X_new[2] < 0:
                    c_status[i] = 0
                else:
                    X_new = np.array([X_new[0], X_new[1], X_new[2]])
                    S['X'] = np.vstack((S['X'], X_new.reshape(1, 3)))
                    S['P'] = np.vstack((S['P'], S['C'][i]))

        # remove triangulated candidates from the state
        S['C'] = S['C'][c_status == 0]
        S['F'] = S['F'][c_status == 0]
        S['T'] = S['T'][c_status == 0]

    # Extract new candidate keypoints and update state
    candidate_P = extract_candidate_keypoints(new_frame, S['P'], S['C'])
    print("\n---EXTRACTING NEW CANDIDATES---")
    print("Number of extracted candidate keypoints: {}".format(len(candidate_P)))

    if S['C'].shape[0] > 0:
        S['C'] = np.vstack((S['C'], candidate_P))
        S['F'] = np.vstack((S['F'], candidate_P))
        S['T'] = np.vstack([S['T'], np.tile(T_current, (candidate_P.shape[0], 1))])
    else:
        S['C'] = candidate_P
        S['F'] = candidate_P
        S['T'] = np.tile(T_current, (candidate_P.shape[0], 1))

    return S, T_current
