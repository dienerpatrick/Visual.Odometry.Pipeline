import cv2
import numpy as np


def track_keypoints(features, frame1, frame2):
    """
    Track features from previous frame to current frame
    :param features: [np.ndarray] features from previous frame
    :param frame1: [np.ndarray] previous frame
    :param frame2: [np.ndarray] current frame
    """

    lk_params = dict(winSize=(20, 20),  # size of the search window at each pyramid level
                     maxLevel=2,  # 0-based maximal pyramid level number
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), minEigThreshold = 1e-4)

    # calculate optical flow
    new_features, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame2, features.astype(np.float32), None, **lk_params)#, minEigThreshold = 5e-2, **lk_params)

    return np.squeeze(new_features), np.squeeze(status)