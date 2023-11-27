### Visual Odometry Pipeline Semester Project at ETH Zurich ###

A Visual Odometry pipeline was built using 3D-to-2D correspondences and a Markovian setup, as well as 2D-to-2D correspondences. The Visual Odometry pipeline with 3D-to-2D correspondences follows the structure: bootstrapping from frame 0 and frame 2, Sequential Structure from Motion using \lstinline{cv2.calcualte_optical_flow} to track the keypoints across frames, Pose Estimation using RANSAC (with EPnP) to exclude outliers and extract the relative pose, candidate keypoints extraction and tracking, candidate keypoint triangulation to bolster keypoint count. A 3D-to-2D VO pipeline that is locally coherent for about 60 frames was implemented. The Visual Odometry pipeline with 2D-to-2D correspondences was successfully implemented creating a globally coherent trajectory.
