Multi-Object Tracking Project
=============================

In this project, you have been assigned the task of fusing measurements from LiDAR and camera sensors, and developing an Extended Kalman filter (EKF) algorithm to track vehicles in 3D point clouds. This project provides a unique opportunity for you to gain practical experience in implementing sensor fusion and tracking techniques, which are integral components in the development of autonomous vehicles.

We have prepared a few hands-on examples for you all to explore, which will help you grasp the concepts we've covered in the lesson more effectively.

In addition to that, we are sharing Dockerfiles with you so you can comfortably develop your algorithms on your own computers. To get started, kindly refer to the [build instructions](./README.md#build-instructions) for a step-by-step guide on how to utilize them properly.

Notice that you can open all jupyter notebooks inside vscode without using the browser.

 ---

Build Instructions.
-------------------

### Installation

#### **Windows**

1. Install Visual Studio Code.
2. Install Dev Container extension of Visual Studio Code.
3. Install WSL (please check that you are using WSL 2).
4. Install [Vcxsrv](https://sourceforge.net/projects/vcxsrv/)
5. Install [Docker for Windows](https://www.docker.com/).

##### **GPU support**

- Install the latest [NVIDIA driver](https://www.nvidia.com/download/index.aspx)

#### **Linux**

1. Install Visual Studio Code.
2. Install remote ssh extension of Visual Studio Code.
3. Install [Docker for Linux](https://docs.docker.com/engine/install/ubuntu/).

##### **GPU support**

- Install the latest NVIDIA driver
- Install [NVIDIA docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Project Set Up

#### **Windows**

**Make sure that docker engine is running**

1. Open **vcxsrv** (xlaunch program).
2. Select **Multiple windows** and next.

![xlaunch](https://user-images.githubusercontent.com/27258035/225385489-d8c58607-d8f9-4e29-96c8-0801a3e1a883.png)

4. Select **Start no client** and next.
5. Mark **Disable access control** (IMPORTANT)

 ![disable](https://user-images.githubusercontent.com/27258035/225386074-df1976a5-6257-4533-997e-d6d770f1335b.png)

Copy the display that appears when you pose the mouse on the taskbar *(example:28147-PC:0.0)*

![display_name](https://user-images.githubusercontent.com/27258035/225386808-374ddbc7-4baa-4bad-9b16-482213a0213b.png)

9. Open the project in VSCode folder (ctrl+k ctrl+o).
10. Open and uncomment the line **"-e", "DISPLAY=\\",** and change the display name with your display name.
11. Open controls palette clicking the green button in the lower left part of vscode

![vscode](https://user-images.githubusercontent.com/27258035/224482432-78a084f5-ef82-4b42-9028-26e23701bf19.png)

13. Select the option, **Reopen in container**.
![image](https://user-images.githubusercontent.com/27258035/224482946-05fd6d80-fb4d-4888-8af1-6c43b3e0bb60.png)

15. Check that display is forwarded by typing xclock on a new terminal inside vscode.
16. Start developing your algorithms.

##### **CPU support**

- go to [.devcontainer/.devcontainer.json](.devcontainer/devcontainer.json#9)
- Uncomment the line "dockerfile": "../Dockerfile\_CPU"
- Comment the line "dockerfile": "../Dockerfile\_GPU"

##### **GPU support**

- go to [.devcontainer/.devcontainer.json](.devcontainer/devcontainer.json#10)
- Uncomment the line **"dockerfile": "../Dockerfile\_GPU"**
- Uncomment the line **"--gpus", "all",**
- Comment the line "dockerfile": "../Dockerfile\_CPU"
- Check you are using cuda by typing ` nvidia-smi` on a vscode terminal.

#### **Linux**

**Make sure that docker engine is running**

1. Open project in VSCode folder (ctrl+k ctrl+o).
2. Open controls panel clicking the green buttom in the lower left part of vs code.
3. Select the option, **Reopen in container**.
4. Start developing your algorithms.

##### **CPU support**

- go to [.devcontainer/.devcontainer.json](.devcontainer/devcontainer.json#9)
- Uncomment the line "dockerfile": "../Dockerfile\_CPU"
- Comment the line "dockerfile": "../Dockerfile\_GPU"

##### **GPU support**

- go to [.devcontainer/.devcontainer.json](.devcontainer/devcontainer.json#10)
- Uncomment the line **"dockerfile": "../Dockerfile\_GPU"**
- Uncomment the line **"--gpus", "all",**
- Comment the line "dockerfile": "../Dockerfile\_CPU"
- Check you are using cuda by typing ` nvidia-smi` on a vscode terminal.

---

Project Instructions
--------------------

This assignment consists of the following two subtasks.

1. **Object detection:** In this part, you are tasked to perform 3D object detection using a Deep-Learning approach based on Lidar data.
2. **Object tracking:** In this part, you are tasked to perform multi-object tracking using an Extended Kalman filter.

### 1 Object Detection

In the object detection section of this project, your task is to extract Lidar data from the provided dataset and detect vehicles within the acquired point cloud. To accomplish this, you'll need to implement the following set of tasks:

- Select a 3D object detection model suitable for your needs.
- Configure the selected model to function within your local setup (this may vary depending on the model chosen).
- Extract point cloud information from the dataset provided.
- Calculate a Bird's Eye View perspective of the point cloud.
- Forward the Bird's Eye View perspective to the selected detection model.
- Extract vehicle data from the perspective obtained.
- Use performance measures to evaluate the model's effectiveness. (The dataset includes 3D detections of multiple vehicles within the point cloud.)

To perform the object detection process, we recommend utilizing the following two models, along with the provided pre-trained weights from the developers:

- [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds (SFA3D)](https://github.com/maudzung/SFA3D)
- [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://github.com/maudzung/Complex-YOLOv4-Pytorch)

By completing these tasks, you'll gain valuable experience in configuring and utilizing 3D object detection models for real-world applications.

### 2 Multi-Object Tracking.

In the Multi-Object Tracking segment of this project, you are required to track the number of annotated vehicles throughout the frames of the provided dataset. To achieve this, you will need to fuse the information acquired from the 3D object detection module with the detections obtained from the camera images. This will involve implementing the following series of tasks:

- Extract information regarding the 2D detections obtained from the camera.
- Create a Kalman Filter module capable of predicting and updating the 3D state of the track.
- Establish a Track module featuring essential methods (initialization, state update, score update, track delete, etc.) and members that will carry out the tracking process (ID, score, KF, state, etc.).
- Establish an Association module that performs association between the stored tracks and the detections present in the current frame (remember gating, association method, matrices, etc.).
- Establish Sensor modules to differentiate the observation model.
- Implement the entire multi-object tracking pipeline, enabling you to track the vehicles throughout the entire provided sequence. Detection -&gt; State-Prediction (Motion model KF) -&gt; Association -&gt; State-Update (Observation model KF).

By performing these tasks, you will acquire valuable experience in multi-object tracking, a crucial aspect of autonomous vehicle development.

---

Dataset
-------

You will have to work on the dataset provided [here](https://drive.google.com/drive/folders/1xctYmMx_NjB2KkgjFUOog6lcb3JgFVn0?usp=share_link).

The dataset provided for this project comprises a sequence of frames depicting our vehicle's journey through Germany. Each frame is presented in protobuf format, which is a cross-platform data format used for serializing structured data. This open-source format, developed by Google, provides a standardized schema for each frame, serialized in the provided files. Each schema contains data collected from multiple cameras and lidars, which can be efficiently parsed and manipulated using Python. The information encoded includes:

### Frame:

- **id** *(string)*: A unique identifier for the frame
- **cameras** *(list of Camera Object)*: contains data from multiple cameras
- **lidars** *(list of Lidar Object)*: contains data from multiple lidars

### Camera:

- **data** *(bytes fiel)*: the raw image data from the camera
- **width** *(int32 field)*: the width of the image in pixels
- **height** *(int32 field)*: the height of the image in pixels
- **depth** *(int32 field)*: the number of channels in the image (e.g. 3 for RGB)
- **T** *(list of float field)*: a list of transformation values that describe the position and orientation of the camera in space (Vehicle->Camera)
- **K** *(list of float field)*: a list of the camera calibration values. (Shape 3x3)
- **D** *(list of float field)*: a list of the Distortion values of the image.
- **pos** *(Position message field)*: a Position enum value that describes the position of the camera relative to the vehicle
- **detections** *(list of CameraDetection field)*: contains a list of CameraDetection messages, each of which represents an object detection within the camera's view
- **timestamp** (float): contains the timestamp in epochs of the time when the image was recorded.

### CameraDetection:

- **id** *(string field)*: a unique identifier for the detected object
- **type** *(ObjectType field)*: an ObjectType enum value that indicates the type of object detected
- **bbox** *(list of float field)*: a list of four float values that represent the bounding box coordinates of the detected object in the image (x0, y0, width, height)

### Lidar:

- **data** *(list of float field)*: a list of raw lidar data points
- **width** *(int32 field)*: the number of horizontal points in the lidar scan
- **height** *(int32 field)*: the number of vertical points in the lidar scan
- **channels** *(int32 field)*: the number of channels in each lidar point (e.g. 3 for x, y, z coordinates)
- **T** *(list of float field)*: a list of transformation values that describe the position and orientation of the lidar in space (Vehicle->Lidar)
- **pos** *(Position message field)*: a Position enum value that describes the position of the lidar relative to the vehicle
- **detections** *(list of LidarDetection field)*: contains a list of LidarDetection messages, each of which represents an object detection within the lidar's range
- **timestamp** (float): contains the timestamp in epochs of the time when the image was recorded.

### LidarDetection:

- **id** *(string field)*: a unique identifier for the detected object
- **type** *(ObjectType enum field)*: an ObjectType enum value that indicates the type of object detected
- **pos** *(repeated float field)*: a list of three float values that represent the x, y, and z coordinates of the center of the detected object in 3D space
- **rot** *(repeated float field)*: a list of three float values that represent the roll, pitch, and yaw angles of the detected object in 3D space
- **scale** *(repeated float field)*: a list of three float values that represent the x, y, and z scale of the detected object in 3D space

### Decode the frames.

For easiness, we have provided you with a code to decode the data. Please check [dataset\_tools.py](./Project/tools/dataset_tools.py) for the implementation of the decoding code. We have also provided you with a [simple example](./Project/Example.ipynb) where you can see how you can parse the data of each frame in python.

### Warning

Kindly take note that there is a lack of perfect synchronization between the cameras and the lidars. In a single frame, we managed to align the camera and image that were taken around the same time. Nevertheless, it is essential to consider the time delay between the lidar and camera while implementing the multi-object tracking algorithm. Timestamp information is available in every sensor message to aid in this regard.

---

Submission Template
-------------------

The submission must be done in a zip file containing the implemented code and a report in a jupyter notebook with the following specifications:

### Project Overview

This section should contain a brief description of the project and what you are trying to achieve.

### Set up

This section should contain a brief description of the steps to follow to run the code. In the jupyter you should implement the code to run your algorithms, so we can see them running on our local machine.

### 3D Object Detection Part

This section should describe the code utilized for 3D object detection, specifically:

- The chosen model and justification for its selection.
- Model configuration.
- Overview of Bird's Eye View (BEV) calculation.
- Performance metrics attained on the provided dataset.
- Runnable code that demonstrates 3D object detection and a visual representation of the results using a visualization library such as Matplotlib.

### Multi-Object Tracking.

This section should explain your code to perform the multi-object tracking. This section must cover:

- A brief explanation of the Kalman Filter for sensor fusion.
- A brief explanation of the whole tracking pipeline.
- Runnable code that demonstrates multi-object tracking and a visual representation of the results using a visualization library such as Matplotlib. (Check the [example](./Project/Example.ipynb) for simple visualization of the tracking)