import cv2
import numpy as np
from scipy.spatial.distance import  cdist
from src.extract_keypoints import extract_keypoints

def extract_candidate_keypoints(frame, keypoints, candidate_keypoints): #TODO: add comparison to candidate keypoints
    """
    Extract candidate keypoints from the frame
    :param frame: [np.ndarray] current frame
    :param keypoints: [np.ndarray] keypoints in state
    :return: candidate keypoints
    """
    pixel_tolerance = 5

    # Extract candidate keypoints
    candidate_P = extract_keypoints(frame)

    print("Number of candidate keypoints: {}".format(len(candidate_P)))
    print("Number of keypoints in state: {}".format(len(keypoints/2)))

    # Get candidate keypoints that are not already in the state
    rounded_P = np.round(candidate_P, 0)
    rounded_candidate_keypoints = np.round(candidate_keypoints, 0)
    rounded_keypoints = np.round(keypoints, 0)

    print("shapes of rounded vecs:", rounded_P.shape, rounded_candidate_keypoints.shape, rounded_keypoints.shape)

    distances=cdist(rounded_P, rounded_keypoints)
    
    candidate_P_mask = distances < pixel_tolerance
    if candidate_P_mask.any():
        rounded_P = rounded_P[~candidate_P_mask.any(axis=1)]
        candidate_P = candidate_P[~candidate_P_mask.any(axis=1)]

    if candidate_keypoints.size > 0:
        distances=cdist(rounded_P, rounded_candidate_keypoints)
        candidate_P_mask = distances < pixel_tolerance
        if candidate_P_mask.any():
            candidate_P = candidate_P[~candidate_P_mask.any(axis=1)]

    print("Number of candidate keypoints not in state: {}".format(len(candidate_P)))

    return candidate_P


