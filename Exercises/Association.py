# imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats.distributions import chi2

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        
    def associate(self, track_list, meas_list):
        # initialize association matrix
        self.association_matrix = np.inf*np.ones((len(track_list),len(meas_list))) 
        
        # loop over all tracks and all measurements to set up association matrix
        for i in range(len(track_list)): 
            track = track_list[i]
            for j in range(len(meas_list)):
                meas = meas_list[j]
                dist = self.Mahalanobis(track, meas, track.P, meas.H)
                # gating
                if dist < 30:
                    self.association_matrix[i,j] = dist
        return self.association_matrix
        
    def Mahalanobis(self, track, meas, P, H):
        # calc Mahalanobis distance
        if meas.kind == 'lidar':
            meas_vec = np.array(meas.data.pos).reshape(3)
        elif meas.kind == 'cam':
            meas_vec = (np.array(meas.data.bbox)[:2]).reshape(2)
        if meas.kind == 'cam':
            p_x, p_y, p_z = track.kf.x[:3]
            gamma = meas_vec - H(p_x, p_y, p_z-2.5,
                         np.array(meas.misc["T"]).reshape((4,4)),np.array(meas.misc["K"]).reshape((3,3))).reshape(2)
            H_j = meas.misc["H_j"](p_x, p_y, p_z-2.5,
                         np.array(meas.misc["T"]).reshape((4,4)),np.array(meas.misc["K"]).reshape((3,3)))
            S = H_j@P@H_j.transpose()
        else:
            gamma = meas_vec - (H@track.kf.x).reshape(3)
            S = H@P@H.transpose()
        MHD = gamma@np.linalg.inv(S)@gamma.T # Mahalanobis distance formula
        return MHD