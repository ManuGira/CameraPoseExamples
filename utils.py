import numpy as np
import cv2


def house_model3d():
    pts3d = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1,0.5,1.5],
        [0,0.5,1.5],
    ], dtype=float)
    polyedges = [[0, 1, 2, 3, 0], [4, 5, 8, 6, 7, 9, 4], [0, 4], [1, 5], [2, 6], [3, 7], [8, 9]]
    return pts3d, polyedges


def cube_model3d():
    pts3d = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], dtype=float) * target.shape[0]
    polyedges = [[0, 1, 2, 3, 0, 4, 5, 6, 7, 4], [1, 5], [2, 6], [3, 7]]
    return pts3d, polyedges


def resize(img, max_width, max_height=None):
    if max_height is None:
        max_height = max_width
    H, W = img.shape[0:2]
    f = min(max_width / W, max_height / H)
    out = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return out


def camera_intrinsics_matrix(focal_length_mm, hfov_degree, width, height):
    sensor_size_mm_x = np.tan(hfov_degree * 0.5 * np.pi/180)*2*focal_length_mm
    sensor_size_mm_y = sensor_size_mm_x*height/width

    fx = width*focal_length_mm/sensor_size_mm_x
    fy = height*focal_length_mm/sensor_size_mm_y
    cx = width/2
    cy = height/2

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])


def load_image_from_video(video_capture_object):
    read_correctly, img = video_capture_object.read()
    if not read_correctly:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def show_matches_bf(img1, kp1, img2, kp2, matches):
    # https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    # make img3 fit in 1920x1080
    img3 = resize(img3, 1080)

    cv2.imshow("matches bf", img3)
    cv2.waitKey(1)
    # plt.imshow(img3)
    # plt.show()


def show_matches_knn(img1, kp1, img2, kp2, matches):
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, m in enumerate(matches):
        if len(m) < 2:
            continue
        if m[0].distance < 0.7 * m[1].distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    img3 = resize(img3, 1080)
    cv2.imshow("matches knn", img3)
    # cv2.waitKey(1)