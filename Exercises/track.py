import numpy as np

class Track():
    """
    A class that represents a track.
    
    contains the state of the track
    """

    def __init__(self, 
                 init_state = None,
                 priors = None,
                 var_priors = None,
                 posteriors = None,
                 var_posteriors = None
                ):
        self.state = []
        self.P = np.eye(4,4)
        self.priors = priors
        self.var_priors = var_priors
        self.posteriors = posteriors
        self.var_posteriors = var_posteriors
    
        if type(init_state) == np.ndarray:
            self.state.append(init_state)

    def add_state(self, state):
        self.state.append(state)
        
