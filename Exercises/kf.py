import numpy as np


class KalmanFilter():
    def __init__(self, x, dim_state=1, q=4, P=None, F=None,):
        self.x = x.reshape(6, 1)  # state
        self.P = P  # uncertainty covariance
        self.F_static = F[0]  # state transition matrix
        self.F_dynamic = F[1]  # state transition matrix
        self.q = q  # process noise

    def get_F(self, dt):
        """
        This method returns the state transition matrix for a given time step dt
        """
        F = self.F_static + self.F_dynamic * dt
        return F

    def get_Q(self, dt):
        """
        This method returns the process noise covariance matrix for a given time step dt
        """
        q = self.q
        return np.array(
            [
                [1/3 * q * dt**3, 0, 0, 1/2 * q * dt**2, 0, 0],
                [0, 1/3 * q * dt**3, 0, 0, 1/2 * q * dt**2, 0],
                [0, 0, 1/3 * q * dt**3, 0, 0, 1/2 * q * dt**2],
                [1/2 * q * dt**2, 0, 0, q * dt, 0, 0],
                [0, 1/2 * q * dt**2, 0, 0, q * dt, 0],
                [0, 0, 1/2 * q * dt**2, 0, 0, q * dt]
            ]
        )

    def predict(self, dt):
        """
        This method predicts the next state by advancing the time step
        In our simple system we have no known external influence (u_k)

        Explanation of variables:
        x: state vector
            x is the state vector of the system
        P: uncertainty covariance
            P is the covariance matrix of the state vector
            It describes the uncertainty of the state vector (gaussian distribution)
        Q: process noise covariance
            additional uncertainty from the environment
        """

        F = self.get_F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.get_Q(dt)

    def update(self, z, R, H):
        """
        z: measurement vector
            can be either from the camera observation (x,y) or from the lidar sensor (x,y,z)
        R: measurement noise covariance
            R is the covariance matrix of the measurement noise
        H: measurement function
            H maps the state space into the measurement space

        R and H need to be supplied by the caller, because they depend on the sensor type
        """

        # eq. 19 from bzarg.com/p/how-a-kalman-filter-works-in-pictures/
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)

        # update state estimate (first part of eq. 18)
        self.x = self.x + K @ (z - (H @ self.x).reshape(3)).reshape(3, 1)
        # second part of eq. 18
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P
