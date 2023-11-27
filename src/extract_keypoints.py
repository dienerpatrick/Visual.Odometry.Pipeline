import cv2
import numpy as np


def extract_keypoints(frame):
    """
    Establish keypoints for the first frame of the dataset
    :param dataset: [string] "parking", "malaga" or "kitti"
    :param frame: [np array] image to extract keypoints from
    :return: keypoints
    """

    feature_params = dict(maxCorners=1000, # maximum number of corners to be returned
                          qualityLevel=0.1, # minimum accepted quality of image corners
                          minDistance=10, # minimum possible Euclidean distance between the returned corners
                          blockSize=3) # size of an average block for computing a derivative covariation matrix

    features0 = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)

    return np.squeeze(features0)
