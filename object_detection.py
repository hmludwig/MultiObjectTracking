import argparse
import sys
import os
import time
import warnings
import zipfile
import torch

from data_process.kitti_bev_utils import makeBEVMap

###################################################################
import matplotlib.pyplot as plt

from PIL import Image
import open3d as o3d

import tools.dataset_tools as dataset_tools
from tools.frame_pb2 import Frame
import tools.plot_tools as plot_tool

from os import listdir
from os.path import isfile, join
from natsort import natsorted, ns

from typing import Tuple
from easydict import EasyDict as edict

import numpy as np
from numpy.lib.function_base import percentile
import numpy.lib.recfunctions as rf


def get_frame_pcl(frame_path):
    frame = dataset_tools.read_frame(frame_path)
    lidar = frame.lidars[0]
    return dataset_tools.decode_lidar(lidar)


def get_frame_lidar(frame_path):
    frame = dataset_tools.read_frame(frame_path)
    lidar = frame.lidars[0]
    return lidar.detections


def get_filtered_lidar(lidar, boundary, labels=None):
    minX = boundary['minX']
    maxX = boundary['maxX']
    minY = boundary['minY']
    maxY = boundary['maxY']
    minZ = boundary['minZ']
    maxZ = boundary['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX)
                    & (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY)
                    & (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
    lidar = lidar[mask]
    lidar[:, 2] = lidar[:, 2] - minZ

    if labels is not None:
        label_x = (labels[:, 1] >= minX) & (labels[:, 1] < maxX)
        label_y = (labels[:, 2] >= minY) & (labels[:, 2] < maxY)
        label_z = (labels[:, 3] >= minZ) & (labels[:, 3] < maxZ)
        mask_label = label_x & label_y & label_z
        labels = labels[mask_label]
        return lidar, labels
    else:
        return lidar


def get_frame_img(frame_path):
    frame = dataset_tools.read_frame(frame_path)
    camera = frame.cameras[0]
    img = dataset_tools.decode_img(camera)
    image = Image.fromarray(img)
    return image


def get_xyzi_points(cloud_array, remove_nans=False):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
	a 3xN matrix.
    '''
    # remove crap points
    # print(cloud_array.dtype.names)
    if remove_nans:
        mask = np.isfinite(cloud_array[:, 0]) & np.isfinite(
            cloud_array[:, 1]) & np.isfinite(cloud_array[:, 2]) & np.isfinite(
                cloud_array[:, 3])
        cloud_array = cloud_array[mask]

    # pull out x, y, and z values + intensty
    points = np.zeros(cloud_array.shape)
    points[..., 0] = cloud_array[:, 0]
    points[..., 1] = cloud_array[:, 1]
    points[..., 2] = cloud_array[:, 2]
    points[..., 3] = cloud_array[:, 3]

    return points


OUSTER_CHANNELS = [
    'x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'ambient', 'range'
]


def pcl_to_bev(pcl: np.ndarray, configs: edict) -> np.ndarray:
    """Computes the bev map of a given pointcloud. 
    
    For generality, this method can return the bev map of the available 
    channels listed in '''BEVConfig.VALID_CHANNELS'''. 

    Parameters
    ----------
        pcl (np.ndarray): pointcloud as a numpy array of shape [n_points, m_channles] 
        configs (Dict): configuration parameters of the resulting bev_map

    Returns
    -------
        bev_map (np.ndarray): bev_map as numpy array of shape [len(config.channels), configs.bev_height, configs.bev_width ]
    """

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((pcl[:, 0] >= configs.lims.x[0])
                    & (pcl[:, 0] <= configs.lims.x[1])
                    & (pcl[:, 1] >= configs.lims.y[0])
                    & (pcl[:, 1] <= configs.lims.y[1])
                    & (pcl[:, 2] >= configs.lims.z[0])
                    & (pcl[:, 2] <= configs.lims.z[1]))
    pcl = pcl[mask]

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    pcl[:, 2] = pcl[:, 2] - configs.lims.z[0]

    # Convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    # compute bev-map discretization by dividing x-range by the bev-image height
    bev_x_discret = (configs.lims.x[1] -
                     configs.lims.x[0]) / configs.bev_height
    bev_y_discret = (configs.lims.y[1] - configs.lims.y[0]) / configs.bev_width
    ## transform all metrix x-coordinates into bev-image coordinates
    pcl_cpy = np.copy(pcl)
    pcl_cpy[:, 0] = np.int_(np.floor(pcl_cpy[:, 0] / bev_x_discret))
    # transform all y-coordinates making sure that no negative bev-coordinates occur
    pcl_cpy[:, 1] = np.int_(
        np.floor(pcl_cpy[:, 1] / bev_y_discret) + (configs.bev_width + 1) / 2)
    # Create BEV map
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    # Compute height and density channel
    pcl_height_sorted, counts = sort_and_map(pcl_cpy, 2, return_counts=True)
    xs = np.int_(pcl_height_sorted[:, 0])
    ys = np.int_(pcl_height_sorted[:, 1])
    # Fill height map
    normalized_height = pcl_height_sorted[:, 2] / float(
        np.abs(configs.lims.z[1] - configs.lims.z[0]))
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    height_map[xs, ys] = normalized_height

    # Fill density map
    normalized_density = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    density_map[xs, ys] = normalized_density

    # Compute intesity channel
    pcl_cpy[pcl_cpy[:, 3] > configs.lims.intensity[1],
            3] = configs.lims.intensity[1]
    pcl_cpy[pcl_cpy[:, 3] < configs.lims.intensity[0],
            3] = configs.lims.intensity[0]

    pcl_int_sorted, _ = sort_and_map(pcl_cpy, 3, return_counts=False)
    xs = np.int_(pcl_int_sorted[:, 0])
    ys = np.int_(pcl_int_sorted[:, 1])
    normalized_int = pcl_int_sorted[:, 3] / (np.amax(pcl_int_sorted[:, 3]) -
                                             np.amin(pcl_int_sorted[:, 3]))
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    intensity_map[xs, ys] = normalized_int

    # Fill BEV
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]

    return bev_map


def sort_and_map(pcl: np.ndarray,
                 channel_index: int,
                 return_counts: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Function to re-arrange elements in poincloud by sorting first by x, then y, then -channel.
    This function allows users to map a pointcloud channel to a top view image (in z axis) of that channel.

    Parameters
    ----------
        pcl (np.ndarray): Input pointcloud of of shape [n_points, m_channles]
        channel_index (int): Index of channel to take into account as third factor, 
                             when sorting the pointcloud.
        return_counts (bool): True to return the counts on points per cell. Used for density channel
    Returns
     ----------
       channel_map (np.ndarray): [description]
       counts (np.ndarray): [description]
       
    """

    idx = np.lexsort((-pcl[:, channel_index], pcl[:, 1], pcl[:, 0]))
    pcl_sorted = pcl[idx]
    counts = None
    # extract all points with identical x and y such that only the maximum value of the channel is kept
    if return_counts:
        _, indices, counts = np.unique(pcl_sorted[:, 0:2],
                                       axis=0,
                                       return_index=True,
                                       return_counts=return_counts)
    else:
        _, indices = np.unique(pcl_sorted[:, 0:2], axis=0, return_index=True)
    return (pcl_sorted[indices], counts)


def show_bev_map(bev_map: np.ndarray) -> None:
    """Function to show bev_map as an RGB image

    By default, the image will only show the 3 first channels of `bev_map`. 

    Parameters
    ----------
        bev_map (np.ndarray): bev_map as numpy array of shape `[len(config.channels), configs.bev_height, configs.bev_width ]` 
    """
    bev_image: np.ndarray = (np.swapaxes(np.swapaxes(bev_map, 0, 1), 1, 2) *
                             255).astype(np.uint8)
    mask: np.ndarray = np.zeros_like(bev_image[:, :, 0])

    height_image = Image.fromarray(np.dstack((bev_image[:, :, 0], mask, mask)))
    den_image = Image.fromarray(np.dstack((mask, bev_image[:, :, 1], mask)))
    int_image = Image.fromarray(np.dstack((mask, mask, bev_image[:, :, 2])))

    int_image.show()
    den_image.show()
    height_image.show()
    Image.fromarray(bev_image).show()


configs = edict()
configs.lims = edict()
configs.lims.x = [0, 50]
configs.lims.y = [-25, 25]
configs.lims.z = [-1.5, 3]
configs.lims.intensity = [0, 100.0]
configs.bev_height = 640
configs.bev_width = 640

mypath = "./data_2/"
onlyfiles = natsorted(
    [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))],
    key=lambda y: y.lower())
lidar_detections = [get_frame_lidar(f) for f in onlyfiles]
bev_data = [pcl_to_bev(get_frame_pcl(f), configs) for f in onlyfiles]
img_data = [get_frame_img(f) for f in onlyfiles]

###################################################################

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.demo_dataset import Demo_KittiDataset
from models.model_utils import create_model
from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration
from utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit
from utils.torch_utils import _sigmoid

if __name__ == '__main__':
    configs = parse_demo_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(
        configs.pretrained_path)
    model.load_state_dict(
        torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device(
        'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()

    out_cap = None

    demo_dataset = Demo_KittiDataset(configs)

    frame_det = {}
    with torch.no_grad():
        for i, (img, bev_map) in enumerate(zip(img_data, bev_data)):

            bev_map = torch.from_numpy(bev_map).to(configs.device,
                                                   non_blocking=True).float()

            detections, bev_map, fps, precious = do_detect(configs,
                                                           model,
                                                           bev_map,
                                                           is_front=True)

            # Draw prediction in the image
            bev_map = (bev_map.permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8)

            bev_map = cv2.resize(bev_map, (640, 640))
            bev_map = draw_predictions(bev_map, detections,
                                       configs.num_classes)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            img_bgr = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)
            calib = Calibration(configs.calib_path)
            kitti_dets = convert_det_to_real_values(detections)

            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:])

                img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

            out_img = merge_rgb_to_bev(img_bgr,
                                       bev_map,
                                       output_width=configs.output_width)
            frame_det[i] = (detections, bev_map)

            if out_cap is None:
                out_cap_h, out_cap_w = out_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out_path = os.path.join(
                    configs.results_dir,
                    '{}_front.avi'.format(configs.foldername))
                print('Create video writer at {}'.format(out_path))
                out_cap = cv2.VideoWriter(out_path, fourcc, 30,
                                          (out_cap_w, out_cap_h))

            out_cap.write(out_img)

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()

import pickle
with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(frame_det, f)
