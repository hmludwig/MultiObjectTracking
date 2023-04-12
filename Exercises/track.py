import numpy as np

class Track():
    """
    A class that represents a track.
    
    contains the state of the track
    """

    def __init__(self, init_state = None):
        self.state = []
        if type(init_state) == np.ndarray:
            self.state.append(init_state)

    def add_state(self, state):
        self.state.append(state)
        
