from google.protobuf import message
import numpy as np

from .frame_pb2 import Frame, Camera, Lidar


def read_frame(path:str) -> Frame:
    frame = Frame()
    frame_data = None
    
    with open(path, "rb") as f:
        # Read the file contents as bytes
        frame_data = f.read()
    try:
        frame.ParseFromString(frame_data)
    except:
        raise message.DecodeError(f"{path} is not a valid frame")

    return frame

def decode_img(camera: Camera) -> np.array:
    
    if type(camera) != Camera:
        raise TypeError(f"There is no conversion from {type(camera)} to 'Camera' Object")
    
    img = np.frombuffer(camera.data, dtype=np.uint8)
    return np.reshape(img, (camera.width, camera.height, camera.depth))


def decode_lidar(lidar:Lidar) -> np.array:

    if type(lidar) != Lidar:
        raise TypeError(f"There is no conversion from {type(lidar)} to {type(Lidar)} Object")
    
    pcl = np.asarray(lidar.data)
    return np.reshape(pcl,(lidar.width*lidar.height, lidar.channels))