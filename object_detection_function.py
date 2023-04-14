import sys
import os
import warnings
import pickle
from os import listdir
from typing import Tuple
from os.path import isfile, join
from natsort import natsorted

import cv2
import torch
import numpy as np
from PIL import Image
from easydict import EasyDict as edict

from SFA3D.sfa.data_process.kitti_bev_utils import makeBEVMap
import tools.dataset_tools as dataset_tools
from tools.frame_pb2 import Frame
import tools.plot_tools as plot_tool

from SFA3D.sfa.models.model_utils import create_model
from SFA3D.sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import SFA3D.sfa.config.kitti_config as cnf
from SFA3D.sfa.data_process.transformation import lidar_to_camera_box
from SFA3D.sfa.utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from SFA3D.sfa.data_process.kitti_data_utils import Calibration
from SFA3D.sfa.utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit
from SFA3D.sfa.utils.torch_utils import _sigmoid

warnings.filterwarnings("ignore", category=UserWarning)


def get_frame_pcl(frame_path):
    frame = dataset_tools.read_frame(frame_path)
    lidar = frame.lidars[0]
    return dataset_tools.decode_lidar(lidar)


def get_frame_img(frame_path):
    frame = dataset_tools.read_frame(frame_path)
    camera = frame.cameras[0]
    img = dataset_tools.decode_img(camera)
    image = Image.fromarray(img)
    return image


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



def object_detection(DATA_PATH = "./data_2/"):
    
    configs = edict()
    configs.lims = edict()
    configs.lims.x = [0, 50]
    configs.lims.y = [-25, 25]
    configs.lims.z = [-1.5, 3]
    configs.lims.intensity = [0, 100.0]
    configs.bev_height = 640
    configs.bev_width = 640

    onlyfiles = natsorted(
        [DATA_PATH + f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))],
        key=lambda y: y.lower())
    data = [(pcl_to_bev(get_frame_pcl(f), configs), get_frame_img(f)) for f in onlyfiles]
    
    src_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(
        str(src_dir + '/SFA3D/sfa/')
    )
    
    
    configs = parse_demo_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    pretrained_path = src_dir + '/SFA3D/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth'
    assert os.path.isfile(pretrained_path), "No file at {}".format(
        configs.pretrained_path)
    model.load_state_dict(
        torch.load(pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device(
        'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()

    out_cap = None

    frame_det = {}
    print('aqui')
    with torch.no_grad():
        for i, (bev_map, img) in enumerate(data):

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
            calib = Calibration(src_dir + '/SFA3D/dataset/kitti/demo/calib.txt')
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
                out_path= './output_video.avi'
                print('Create video writer at {}'.format(out_path))
                out_cap = cv2.VideoWriter(out_path, fourcc, 30,
                                          (out_cap_w, out_cap_h))

            out_cap.write(out_img)

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()

    with open('./output_data.pkl', 'wb') as f:
        pickle.dump(frame_det, f)
    return frame_det
