import numpy as np

class Camera:
    '''Camera sensor class including field of view and coordinate transformation'''
    def __init__(self, camera_gt):
        self.fov = [-np.pi/4, np.pi/4] # sensor field of view / opening angle

        # coordiante transformation matrix from sensor to vehicle coordinates
        self.sens_to_veh = camera_gt.T           
        self.veh_to_sens = np.linalg.inv(self.sens_to_veh) # transformation vehicle to sensor coordinates
    
    def in_fov(self, x):
        # check if an object x can be seen by this sensor
        pos_veh = np.ones((4, x.shape[1])) # homogeneous coordinates
        pos_veh[0:3,:] = x[0:3,:] 
        pos_sens = self.veh_to_sens@pos_veh # transform from vehicle to sensor coordinates
        alpha = np.arctan(pos_sens[1]/pos_sens[0]) # calc angle between object and x-axis
        # no normalization needed because returned alpha always lies between [-pi/2, pi/2]
        return ((alpha > self.fov[0]) & (alpha < self.fov[1]) & (pos_sens[0]<0)), alpha