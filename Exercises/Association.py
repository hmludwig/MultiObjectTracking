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
                dist = self.Mahalanobis(track, meas, track.P)
                # gating
                if dist < chi2.ppf(0.95, df=2):
                    self.association_matrix[i,j] = dist
        return self.association_matrix
        
    def Mahalanobis(self, track, meas, P):
        # calc Mahalanobis distance
        H = np.matrix([[1, 0, 0, 0],
                       [0, 1, 0, 0]]) 
        gamma = meas - H@track.state[-1]
        S = H@P@H.transpose()
        MHD = gamma@np.linalg.inv(S)@gamma.T # Mahalanobis distance formula
        return MHD