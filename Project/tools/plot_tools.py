import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from . import image_tools

COLOR_DICT = {
    "confirmed" : "green",
    "tentative" : "yellow",
    "initialized" : "red"
}

LABEL_DICT = {
    "confirmed" : "Confirmed track",
    "tentative" : "Tentative track",
    "initialized" : "Initialized track",
}

def plot_tracks(img, tracks, measurements, lidar_detections, camera, state=None):
    fig, (ax1, ax2) = plt.subplots(1,2)

    camera_mtx = np.array(camera.K).reshape(3,3)
    dist = np.array(camera.D)
    img1, img2 = image_tools.undistort(img, camera_mtx, dist)

    T = np.asarray(camera.T)
    T = np.reshape(T, (4,4))
    ax2.imshow(img1)

    for track in tracks:
        if state == None or track.state == state:
            color = COLOR_DICT[track.state]
            plt_label = LABEL_DICT[track.state]
            
            # get current track state
            w = track.w
            h = track.h
            l = track.l

            x0 = track.x[0] - l/2
            y0 = track.x[1] + w/2
            z0 = track.x[2] - h/2
            angle = track.yaw

            # plot bbox of track in bird eye view
            bbox = plt.Rectangle((-y0,x0),w,l, color=color, angle=angle, alpha=0.2)
            ax1.add_patch(bbox)

            # plot track position
            ax1.text(-track.x[1], track.x[0]+1, track.id)
            ax1.scatter(-track.x[1], track.x[0], color=color, marker="x", label = plt_label)

            # plot bbox on image
            # project veh pos on image
            veh_pos = np.ones((3,1))
            veh_pos[0:3,0] = track.x[0:3]

                        
            # bounding box corners
            x_corners = [-l/2, l/2, l/2, l/2, l/2, -l/2, -l/2, -l/2]  
            y_corners = [-w/2, -w/2, -w/2, w/2, w/2, w/2, w/2, -w/2]  
            z_corners = [-h/2, -h/2, h/2, h/2, -h/2, -h/2, h/2, h/2]  

            # bounding box
            corners_3D = np.array([x_corners, y_corners, z_corners])

            # translate
            corners_3D += veh_pos

            # translate
            homogeneous_coord = np.ones((corners_3D.shape[0]+1, corners_3D.shape[1]))
            homogeneous_coord[:3,:] = corners_3D

            #transform corners to camera frame of reference
            corners_3D = np.dot(T,homogeneous_coord)
            corners_3D = corners_3D[:3,:]
            
            depth = corners_3D[-1:]
            scaled_corners_3D = corners_3D / depth


            newcammtx, x, y, w, h = image_tools.get_offsets(img, camera_mtx, dist)  # getting offsets and new camera matrix
            corners_3D = np.dot(newcammtx, scaled_corners_3D).T  # rotation to fit the pointcloud with pixels

            ch_offset_pcd = []  # changing the offset of the pointcloud to match the cropped undistorted image
            for row in corners_3D:
                ch_offset_pcd.append([row[0]-x, row[1]+y, row[2]])
            ch_offset_pcd = np.array(ch_offset_pcd)
            ch_offset_pcd[:, -1] = depth


            cropped_pcd = ch_offset_pcd  # cropping the pointcloud so that only points within the image remain
            cropped_pcd = cropped_pcd.T

            # remove bounding boxes that include negative x, projection makes no sense
            if np.any(corners_3D[2,:] <= 0):
                continue
            
            # project to image
            corners_2D = np.zeros((2,8))
            corners_2D[:2,:] = cropped_pcd[:2,:]
            draw_line_indices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

            paths_2D = np.transpose(corners_2D[:, draw_line_indices])
            
            codes = [Path.LINETO]*paths_2D.shape[0]
            codes[0] = Path.MOVETO
            path = Path(paths_2D, codes)
                
            # plot bounding box in image
            p = patches.PathPatch(
                path, fill=False, color=color, linewidth=3)
            ax2.add_patch(p)


    # plot groud truth positions of vehicles
    for detection in lidar_detections:
        lx = detection.pos[0]
        ly = -detection.pos[1]
        ax1.scatter(ly, lx, color="gray", s=80, marker='+', label="ground truth")

    #Axes configurations
    ax1.set_xlabel('y [m]')
    ax1.set_ylabel('x [m]')
    ax1.set_aspect('equal')
    ax1.set_ylim(0, 50) 
    ax1.set_xlim(-10, 10)