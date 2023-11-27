import cv2
import numpy as np
from src.utils import projectPoints, estimatePoseDLT

NO_RANSAC=False
NO_DLT=True


def ransacLocalization(keypoints, landmarks, K):
    num_iterations = float('inf')
    pixel_tolerance = 10
    min_inlier_count = 15   # TODO: Choose the right inlier_count (was 30)
    k = 3                   # TODO: Choose the right k (was 4)
    confidence = 0.95
    upper_bound_on_outlier_ratio = 0.9
    
    # cap the number of iterations at 15000
    num_iterations = 15000

    # Initialize RANSAC
    best_inlier_mask = np.zeros(keypoints.shape[1])
    keypoints = np.flip(keypoints, axis=0)
    max_num_inliers = 0

    # RANSAC
    i = 0
    while num_iterations > i:
        # Model from k samples (DLT or P3P)
        if NO_RANSAC:
            break
        indices = np.random.permutation(landmarks.shape[0])[:k]
        landmark_sample = landmarks[indices, :]
        keypoint_sample = keypoints[:, indices]

        success, rotation_vectors, translation_vectors = cv2.solveP3P(landmark_sample, keypoint_sample.T, K,
                                                                      None, flags=cv2.SOLVEPNP_P3P)
        
        if success:
            t_C_W_guess = []
            R_C_W_guess = []
            for rotation_vector in rotation_vectors:
                rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
                for translation_vector in translation_vectors:
                    R_C_W_guess.append(rotation_matrix)
                    t_C_W_guess.append(translation_vector)
            # Count inliers

            is_inlier = np.zeros(landmarks.shape[0])
            best_index = 0
            for alt_idx in range(len(R_C_W_guess)):
                C_landmarks = np.matmul(R_C_W_guess[alt_idx], landmarks[:, :, None]).squeeze(-1) + \
                            t_C_W_guess[alt_idx][None, :].squeeze(-1)
                projected_points = projectPoints(C_landmarks, K)
                difference = keypoints - projected_points.T
                errors = (difference ** 2).sum(0)
                alternative_is_inlier = errors < pixel_tolerance ** 2
                if alternative_is_inlier.sum() > is_inlier.sum():
                    is_inlier = alternative_is_inlier
                    best_index = alt_idx

            if is_inlier.sum() > max_num_inliers and is_inlier.sum() >= min_inlier_count:
                max_num_inliers = is_inlier.sum()
                # print(" Amount of mbest inliers: ", is_inlier.sum())
                best_inlier_mask = is_inlier

            # estimate of the outlier ratio
            outlier_ratio = 1 - max_num_inliers / is_inlier.shape[0]            
            outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
            num_iterations = np.log(1 - confidence) / np.log(1 - (1 - outlier_ratio) ** k)
            num_iterations = min(15000, num_iterations)
        
        i += 1

    print(f"RANSAC stopped after {i} iterations")

    if max_num_inliers == 0:
        print("No inliers found or FuckRansac is True")
        M_C_W  = estimatePoseDLT(keypoints[:, :].T, landmarks[:, :], K)
        R_C_W = M_C_W[:, :3]
        t_C_W = M_C_W[:, -1]
    else:
        if NO_DLT:
            
            print("best P3P returns:",success,R_C_W_guess[best_index],t_C_W_guess[best_index])
            
            return R_C_W_guess[best_index],t_C_W_guess[best_index], best_inlier_mask
        else:
            M_C_W  = estimatePoseDLT(keypoints[:, best_inlier_mask].T, landmarks[best_inlier_mask, :], K)
            R_C_W = M_C_W[:, :3]
            t_C_W = M_C_W[:, -1]

    return R_C_W, t_C_W, best_inlier_mask

