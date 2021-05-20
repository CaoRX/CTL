import CTL.funcs.funcs as funcs
from CTL.funcs.stringSet import StringSet
from copy import deepcopy
import warnings

class TensorBase:
    # attributes: xp, a

    def __init__(self, data = None):
        self.a = data

    @property 
    def dim(self):
        if (self.a is None):
            print("Error: dim is asked while TensorBase has not been initialized yet.")
            return None
        return len(self.a.shape)
    
    @property
    def shape(self):
        if (self.a is None):
            print("Error: dim is asked while TensorBase has not been initialized yet.")
            return None
        return self.a.shape









