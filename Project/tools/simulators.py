import numpy as np

class CarSimulator(object):
    def __init__(self, x0=0.0, vel=1.0, meas_std=0.0, process_std=0.0):
        self.x=np.copy(x0)
        self.x = self.x.astype(float)

        self.vel=np.copy(vel)
        self.vel = self.vel.astype(float)

        self.meas_std = np.copy(meas_std)
        self.meas_std = self.meas_std.astype(float)

        self.process_std = np.copy(process_std)
        self.process_std = self.process_std.astype(float)
        self.state_dim = self.x.shape[0] if len(self.x.shape) > 1 else 1

    def move(self, dt=1.0):
        dx = self.vel+np.random.randn()*self.process_std
        self.x += dx*dt
        return np.copy(self.x)

    def measure_pos(self):
        return np.copy(self.x + np.random.randn()*self.meas_std)
    
    def simulate_steps(self, N=5, dt=1):
        
        gts = np.zeros((N,self.state_dim))
        measurements = np.zeros((N,self.state_dim))

        for i in range(N):
            gts[i,:] = self.move(dt).flatten()
            measurements[i,:] = self.measure_pos().flatten()

        return gts, measurements


if __name__ == '__main__':
    import math

    N = 25

    pos_x = 0.0
    pos_y = 0.0
    vx = 1.3
    vy = 2.1

    sigma_x = 2
    sigma_y = 3

    meas_sig_x = 4.5
    meas_sig_y = 3.5

    x0 = np.array([[pos_x],
                [pos_y],
                [vx],
                [vy]])

    process_var = np.array([[sigma_x**2],
                            [sigma_y**2]])

    sensor_var = np.array([[meas_sig_x],
                        [meas_sig_y]])

    process_std = np.sqrt(process_var)
    sensor_std = np.sqrt(sensor_var)

    car = CarSimulator(x0=x0[:2], vel=x0[2:], process_std=process_std, meas_std=sensor_std)
    gts, zs = car.simulate_steps(N)
