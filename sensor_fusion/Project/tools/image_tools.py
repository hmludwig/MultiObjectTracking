import cv2
import numpy as np

def translate(arr: np.array, os_transl: list) -> np.array:
    """
    Takes the pointclouds as numpy arrays and translates the points to fit to the center lidar.
    :param arr: numpy array of a pointcloud
    :param os_transl: list of numbers specifying the shift in x, y, z direction
    :return: translated numpy array of pointcloud
    """
    new_arr = np.zeros_like(arr, dtype=np.float32)
    for i, row in enumerate(arr):
        new_arr[i, :] = row + os_transl[i]
    return new_arr

def get_offsets(img, camera_mtx: np.array, dist: np.array):
    """
    Returns the values, which were used for undistorting and cropping the image.
    This is used to get a good new camera_matrix and to crop the pointcloud to fit into the undistorted image.
    :param image: image name as string
    :param camera_mtx: camera matrix from the intrinsic.yaml file
    :param dist: distortion coefficients from intrinsic.yaml file
    :return: new camera matrix, x: offset in x direction (width), y: offset in y direction (height), width, height
    """
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_mtx, dist, (w, h), 1, (w, h))
    x, y, w, h = roi

    return newcameramtx, x, y, w, h

def crop(pcd: np.array, max_depth: int, w, h) -> np.array:
    """
    cuts the pointcloud, such that only the points within the image remain,
    and so that the maximum depth corresponds to the depth specified
    :param pcd: array with [row] number of points and x, y, depth columns
    :param img: used to get the width and height of the image
    :param max_depth: points with higher depth than max_depth are cut out, for better visibility
    :return: cropped pcd, where only the points seen in the image remain
    """
    mask = np.where((pcd[:, 0] >= 0) & (pcd[:, 0] <= w)
                    & (pcd[:, 1] >= 0) & (pcd[:, 1] <= h)
                    & (pcd[:, 2] >= 0) & (pcd[:, 2] <= max_depth))  # ignores everything after [int] meters
    return pcd[mask]

def undistort(img, camera_mtx: np.array, dist: np.array):
    """
    undistorts the image according to the distortion coefficients and the camera matrix in intrinsic.yaml file
    :param image: image name as string
    :param camera_mtx:
    :param dist:
    :return: undistorted-uncropped image and undistroted cropped image
    """
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_mtx, dist, (w, h), 1, (w, h))
    undst = cv2.undistort(img, camera_mtx, dist, None, newcameramtx)
    undst_uncropped = undst
    x, y, w, h = roi
    undst = undst_uncropped[y:y + h, x:x + w]

    return undst_uncropped, undst
