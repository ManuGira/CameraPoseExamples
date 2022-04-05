import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils

def extract_features(img, show=False):
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


def load_target(filename):
    target = cv2.imread(filename, 0)

    # Shrink image
    target = utils.resize(target, 1920/2, 1080/2)

    # blur to remove noise
    target = cv2.blur(target, (3, 3))

    return target


def find_matches_flann(descriptors_target, descriptors_frame):
    # https://stackoverflow.com/questions/25018423/opencv-python-error-when-using-orb-images-feature-matching

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_knn = flann.knnMatch(descriptors_target, descriptors_frame, k=2)
    return matches_knn


def find_matches_bruteforce(descriptors_target, descriptors_frame):
    # Brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_target, descriptors_frame)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # keep 20% of the best matches
    matches = matches[:int(len(matches)*0.2)]

    # plt.plot([m.distance for m in matches])
    # plt.show()

    return matches

def exctract_points_from_matches_bf(kp1, kp2, matches):
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # keep only 50% of stronger matches
    matches = matches[:int(len(matches)*0.5)]

    # extract points
    pts1 = [kp1[m.queryIdx].pt for m in matches]
    pts2 = [kp2[m.trainIdx].pt for m in matches]
    return pts1, pts2


def exctract_points_from_matches_knn(kp1, kp2, matches):
    # filter out single matches
    matches = [m for m in matches if len(m) == 2]

    # filter bad matches with ratio test as per Lowe's paper
    matches = [m[0] for m in matches if m[0].distance < 0.7 * m[1].distance]

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = [kp1[m.queryIdx].pt for m in matches]
    pts2 = [kp2[m.trainIdx].pt for m in matches]
    return pts1, pts2


def project_point3d_to_screen(pts3d, rvec, tvec, camera_matrix, frame=None):
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, cameraMatrix=camera_matrix, distCoeffs=None)
    pts2d = np.array([p[0] for p in pts2d]).astype(np.int32)
    return pts2d



def compute_target_pose_relative_to_camera(pts3d_target, pts2d_frame, camera_matrix):
    if len(pts3d_target) < 4:
        return False, None, None, None

    pts_target_3d = [(p[0], p[1], 0) for p in pts3d_target]

    success, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(pts_target_3d), np.array(pts2d_frame), camera_matrix, None)

    RMat3, jacobian_mat = cv2.Rodrigues(rvec)
    RMat = np.eye(4)
    RMat[:3, :3] = RMat3
    TMat = np.eye(4)
    TMat[0:3, 3:4] = np.array(tvec)
    target_pose = TMat.dot(RMat)  # TODO: verify target_pose is the good result

    return success, target_pose, rvec, tvec



def main():
    src_video = 'assets/scene_carpet.mp4'
    src_target = 'assets/target_carpet.jpg'
    hfov_degree = 84
    focal_length_mm = 24

    target = load_target(src_target)
    keypoints_target, descriptors_target = extract_features(target)

    video_object = cv2.VideoCapture(src_video)
    while True:
        cv2.waitKey(1)

        frame = utils.load_image_from_video(video_object)
        if frame is None:
            break
        frame = utils.resize(frame, 1920/2)
        # plt.imshow(frame)
        # plt.show()

        keypoints_frame, descriptors_frame = extract_features(frame)

        if False:
            matches_bf = find_matches_bruteforce(descriptors_target, descriptors_frame)
            utils.show_matches_bf(target, keypoints_target, frame, keypoints_frame, matches_bf)
            pts_target, pts_frame = exctract_points_from_matches_bf(keypoints_target, keypoints_frame, matches_bf)
        else:
            matches_knn = find_matches_flann(descriptors_target, descriptors_frame)
            utils.show_matches_knn(target, keypoints_target, frame, keypoints_frame, matches_knn)
            pts_target, pts_frame = exctract_points_from_matches_knn(keypoints_target, keypoints_frame, matches_knn)

        H, W = frame.shape[:2]
        camera_matrix = utils.camera_intrinsics_matrix(focal_length_mm, hfov_degree, W, H)

        success, target_pose, rvec, tvec = compute_target_pose_relative_to_camera(pts_target, pts_frame, camera_matrix)

        if not success:
            continue

        pts3d, polyedges = utils.house_model3d()
        pts3d *= target.shape[0]/2
        pts3d[:, 0] += target.shape[1]/2
        pts3d[:, 1] += target.shape[0]/2
        pts3d[:, 2] *= -1
        pts2d_screen = project_point3d_to_screen(pts3d, rvec, tvec, camera_matrix, frame)
        lines = []
        for vertices in polyedges:
            line = np.array([pts2d_screen[v] for v in vertices])
            lines.append(line)

        frame = cv2.polylines(frame, lines, True, 255, 3, cv2.LINE_AA)
        frame = utils.resize(frame, 1080)
        cv2.imshow("proj", frame)






if __name__ == '__main__':
    main()