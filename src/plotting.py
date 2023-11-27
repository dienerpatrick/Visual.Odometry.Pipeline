from src.utils import load_init_data, load_image_paths
import matplotlib.pyplot as plt
import numpy as np
import cv2


########################################## plotting video for comparison ##########################################

def render_video(DATASET, NUM_FRAMES):
    """
    Plot the video of the trajectory
    :return:
    """
    image_paths = load_image_paths(DATASET, NUM_FRAMES)
    # print(image_paths)

    # Plot video
    write = True
    img_array = []
    if write:
        for path in image_paths:
            img = cv2.imread(path)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def keypoint_plotting_on_frames(keypoint_pixel_cords, candidate_keypoint_pixel_cords, frames):
    """
    Plot keypoints and candidate keypoints on current frame
    :param keypoint_pixel_cords: [np.ndarray] keypoints in state
    :param candidate_keypoint_pixel_cords: [np.ndarray] candidate keypoints
    :return: tracked keypoints
    """
    for i,frame in enumerate(frames):
        plt.imshow(frame)
        plt.scatter(keypoint_pixel_cords[i][:,0],keypoint_pixel_cords[i][:,1], s= 1, c = 'r')
        plt.scatter(candidate_keypoint_pixel_cords[i][:,0],candidate_keypoint_pixel_cords[i][:,1], s= 1.2, c = 'g')
        plt.show()


def keypoint_plotting_on_frame(keypoint_pixel_cords, candidate_keypoint_pixel_cords, frame):
    """
    Plot keypoints and candidate keypoints on current frame
    :param keypoint_pixel_cords: [np.ndarray] keypoints in state
    :param candidate_keypoint_pixel_cords: [np.ndarray] candidate keypoints
    :return: tracked keypoints
    """

    plt.imshow(frame, cmap= 'gray')
    plt.scatter(keypoint_pixel_cords[:,0],keypoint_pixel_cords[:,1], s= 1, c = 'r')
    plt.scatter(candidate_keypoint_pixel_cords[:,0],candidate_keypoint_pixel_cords[:,1], s= 1.2, c = 'g')
    plt.show()


def keypoint_plotting_3d(keypoint_world_cords):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(-100, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    colors = ['b', 'g', 'y', 'c', 'm', 'k']
    for i, frame in enumerate(keypoint_world_cords):
        ax.scatter(keypoint_world_cords[i][:,0], keypoint_world_cords[i][:,1], keypoint_world_cords[i][:,2], c=colors[i%len(colors)], marker='o')
    plt.show()


def trajectory_plotting_3d(camera_cords):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.axis('equal')
    ax.scatter(camera_cords[:,0], camera_cords[:,2], c='r', marker='o')
    for i in range(len(camera_cords[:,0])):
        ax.text(camera_cords[i,0], camera_cords[i,2],s='%s'%(str(i)))
    plt.show()


