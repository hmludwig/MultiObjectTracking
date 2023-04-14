import numpy as np
from kf import KalmanFilter
import uuid

class Track():
    """
    A class that represents a track.
    
    contains the state of the track
    """

    def __init__(self, 
                init_state = None,
                P = None,
                Q = None,
                frame_id = None,
                camera = None
                ):
        self.state = []
        F_static = np.eye(6)
        F_dynamic = np.zeros((6,6))
        # set upper right 3x3 to identity
        F_dynamic[0:3,3:6] = np.eye(3)
        if P is None:
            self.P = np.array([
                [1, 0, 0, 0.5, 0, 0],
                [0, 1, 0, 0, 0.5, 0],
                [0, 0, 1, 0, 0, 0.5],
                [0.5, 0, 0, 3, 0, 0],
                [0, 0.5, 0, 0, 3, 0],
                [0, 0, 0.5, 0, 0, 3]
            ])
        else:
            self.P = P
        self.kf = KalmanFilter(init_state, dim_state = 6, P=self.P, F=[F_static, F_dynamic])
        self.score = 1.0
        self.scores = []
        self.assigned = False
        self.id = uuid.uuid4()
        self.range = [frame_id, -1]
        self.record = {}
        self.frame_id = frame_id
        self.in_fov = False
        self.camera = camera
    
        if type(init_state) == np.ndarray:
            self.state.append(init_state)
        self.update_record()

        #print(f"Track created with state: {init_state} at {frame_id}") 

    def update(self, z, R, H):
        """
        Update the state of the track using the Kalman Filter
        """
        self.kf.update(z, R, H)
    
    def predict(self, dt, frame_id):
        """
        Predict the state of the track using the Kalman Filter
        """
        self.frame_id = frame_id
        self.kf.predict(dt)
        self.state.append(self.kf.x)
        self.scores.append(self.score)
        self.in_fov = self.check_fov(camera=self.camera)
        self.update_record()

    def update_record(self):
        self.record[self.frame_id] ={
                "state": self.kf.x,
                "score": self.scores,
                "frame_id": self.frame_id,
                "in_fov": self.in_fov
            }
    
    def check_fov(self, camera):
        """
        Check if the track is in the field of view of the camera
        """
        # get the position of the track
        x = self.kf.x[0]
        y = self.kf.x[1]
        z = self.kf.x[2]
        # check if the track is in the field of view
        self.in_fov, alpha = camera.in_fov(np.array([-y,x,z]))
        #print(f"{x}, {-y}, {z}, {alpha=} in fov: {self.in_fov}")
        return self.in_fov
        
# plt.Rectangle((y-5, x-5), 10, 10, ec='r', fill= Fal