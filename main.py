import matplotlib
from src.extract_keypoints import extract_keypoints
from src.track_keypoints import track_keypoints
from src.extract_candidate_keypoints import extract_candidate_keypoints
from src.triangulate_landmarks import triangulate_landmarks
from src.triangulate_landmarks import triangulate_landmarks_2
from src.ransac_localization import ransacLocalization
from src.check_candidate_keypoints import check_candidate_keypoints
from src.linear_triangulation import linearTriangulation
from src.utils import load_init_data, load_image_paths
from src.update_state import update_state
from src.plotting import render_video, keypoint_plotting_3d, trajectory_plotting_3d, keypoint_plotting_on_frames,keypoint_plotting_on_frame
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
import time


sys_args = sys.argv

if str(sys_args[2]) == '2D-2D':
    method_2d_2d = True
else:
    method_2d_2d = False

DATASET = str(sys_args[1])
FRAMES_FOR_INIT = [0, 2]

# number of frames to run VO on
NUM_FRAMES = int(sys_args[3])
ANGLE_THRESHOLD = np.pi/10
_, _, K = load_init_data(DATASET, FRAMES_FOR_INIT)


########################################## initialization phase ##########################################
def run_initialization():
    '''
    Run the initialization phase of the algorithm, returns the initial state S0
    :return:
    '''

    print("========= Initialization Phase =========")

    # Load init data
    image1, image2, _ = load_init_data(DATASET, FRAMES_FOR_INIT)

    # Compute outlier-free point correspondences
    features0 = extract_keypoints(image1)
    features1, status = track_keypoints(features0, image1, image2)

    print("Number of extracted features: ", features0.shape[0])
    print("Number of tracked features that are inliers: ", np.sum(status))

    features0 = features0[status == 1]
    features1 = features1[status == 1]

    # Plot tracking of keypoints
    fig = plt.figure()
    plt.imshow(image2, cmap='gray')
    plt.plot(features0[:, 0], features0[:, 1], 'r.')
    for i in range(len(features0[:,0])):
        plt.plot([features0[i,0], features1[i,0]], [features0[i,1], features1[i,1]], 'g-', linewidth=3)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

    print(f"shape of features0{features0.shape}")
    P, inliers1, inliers2, R_init, T_init = triangulate_landmarks(features0, features1, K)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-50, 50)
    ax.set_ylim3d(-50, 50)
    ax.set_zlim3d(0, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    colors = ['b', 'g', 'y', 'c', 'm', 'k']
    ax.scatter(P[0], P[1], P[2], c=colors[i%len(colors)], marker='o')
    plt.show()
    # Initialize state
    S = {'P': inliers2, 'X': P[:3].transpose(), 'C': np.array([]), 'F': np.array([]), "T": np.array([])}

    return S


def VO2d_2d(imagepath1, imagepath2, itr):
    '''
    extract features and track to next image
    get pose from 2d-2d correspondences
    :return:
    '''

    print(f"========= frame{itr} =========",end="\r")

    # Load init data
    # image1, image2, _ = load_init_data(DATASET, FRAMES_FOR_INIT)
    image1 = cv2.cvtColor(cv2.imread(imagepath1), cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(cv2.imread(imagepath2), cv2.COLOR_BGR2GRAY)

    # Compute outlier-free point correspondences
    features0 = extract_keypoints(image1)
    features1, status = track_keypoints(features0, image1, image2)
    features0 = features0[status == 1]
    features1 = features1[status == 1]

    R_, T_ = triangulate_landmarks_2(features0, features1, K)

    # plot image2 with features1
    plt.imshow(image2, cmap='gray')
    plt.xlim(0, image2.shape[1])
    plt.ylim(image2.shape[0], 0)
    plt.scatter(features1[:, 0], features1[:, 1], c='r', s=1, marker='x')
    plt.title(f"Frame {itr+1} with tracked features", fontsize=10)
    plt.grid(False)
    plt.axis('off')
    plt.savefig('temp/image.png', bbox_inches='tight')
    plt.clf()

    return R_,T_


def run_continuous():
    plt.figure(1)
    points = []
    T_update = np.eye(4)
    image_paths = load_image_paths(DATASET, NUM_FRAMES)

    for frame in range(NUM_FRAMES):
        r,t=(VO2d_2d(image_paths[frame],image_paths[frame+1],frame))
        T_current = np.c_[r,t]
        T_current = np.r_[T_current,[[0,0,0,1]]]
        T_update = T_update@T_current
        points.append(T_update[0:3,3])
        x_coords = [point[0] for point in points]
        z_coords = [point[2]*-1 for point in points]
        plt.xlim(min(min(x_coords), min(z_coords))-10, max(max(x_coords), max(z_coords))+10)
        plt.ylim(min(min(x_coords), min(z_coords))-10, max(max(x_coords), max(z_coords))+10)
        plt.scatter(x_coords, z_coords, c='b', s=3, marker='.')
        plt.title("Trajectory of the Vehicle", fontsize=15)
        plt.xlabel("x", fontsize=15)
        plt.ylabel("z", fontsize=15)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.savefig('temp/trajectory.png')
        plt.clf()


def run_continuous_tracking():
    """
    Run the continuous tracking phase of the algorithm
    :return:
    """
    camera_cords = [0,0,0]
    keypoint_world_cords = []
    keypoint_pixel_cords = []
    candidate_keypoint_pixel_cords = []
    frames = []

    image_paths = load_image_paths(DATASET, NUM_FRAMES)

    S = run_initialization()
    itr = 0
    for i in range(FRAMES_FOR_INIT[1]+1, len(image_paths)):
        new_frame = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2GRAY)
        old_frame = cv2.cvtColor(cv2.imread(image_paths[i-1]), cv2.COLOR_BGR2GRAY)

        print("\n========= Frame: ", i, " =========")
        print("Number of keypoints (P): ", S['P'].shape[0])
        print("Number of landmarks (X): ", S['X'].shape[0])
        print("Number of candidate keypoints (C): ", S['C'].shape[0])
        print("Number of candidate landmarks (F): ", S['F'].shape[0])
        print("Number of candidate poses (T): ", S['T'].shape[0])

        S,T = update_state(S, old_frame, new_frame, K, ANGLE_THRESHOLD)
        T = T.reshape(3,4)

        keypoint_pixel_cords.append(S['P'])
        candidate_keypoint_pixel_cords.append(S['C'])
        camera_cords = np.vstack([camera_cords,[T[0,3],T[1,3],T[2,3]]])
        keypoint_world_cords.append(S['X'])
        frames.append(new_frame)
        # keypoint_plotting_on_frame(S['P'], S['C'], new_frame)

        plt.imshow(new_frame, cmap='gray')
        plt.xlim(0, new_frame.shape[1])
        plt.ylim(new_frame.shape[0], 0)
        plt.scatter(S['P'][:, 0], S['P'][:, 1], c='r', s=1, marker='x')
        plt.title(f"Frame {itr + 1} with tracked features", fontsize=10)
        plt.grid(False)
        plt.axis('off')
        plt.savefig('temp/image.png', bbox_inches='tight')
        plt.clf()

        x_coords = [point[0] for point in camera_cords]
        z_coords = [point[2]*-1 for point in camera_cords]

        plt.xlim(min(min(x_coords), min(z_coords)) - 10, max(max(x_coords), max(z_coords)) + 10)
        plt.ylim(min(min(x_coords), min(z_coords)) - 10, max(max(x_coords), max(z_coords)) + 10)
        plt.scatter(x_coords, z_coords, c='b', s=3, marker='.')
        plt.title("Trajectory of the Vehicle", fontsize=15)
        plt.xlabel("x", fontsize=15)
        plt.ylabel("z", fontsize=15)
        plt.savefig('temp/trajectory.png')
        plt.clf()

        itr += 1
    
    keypoint_plotting_3d(keypoint_world_cords)
    trajectory_plotting_3d(camera_cords)


if __name__ == '__main__':
    if os.path.exists("project.avi") == False:
        print("Video not found, creating video...")
        render_video(DATASET, NUM_FRAMES)

    print("Video found, running algorithm...")
    if method_2d_2d:
        run_continuous()
    else:
        run_continuous_tracking() 

    