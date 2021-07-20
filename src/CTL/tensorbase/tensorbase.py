import CTL.funcs.funcs as funcs
from CTL.funcs.stringSet import StringSet
from copy import deepcopy
import warnings

class TensorBase:
    """
    The base class for tensors.

    Parameters
    ----------
    data : None or ndarray
        The data saved in the tensor.
        For TensorLike case, only None is needed.
        For DiagonalTensor case, will be 1-D tensor representing the main diagonal.
    
    Attributes
    ----------
    a : None or ndarray
        The data.

    Properties
    ----------
    dim : int
        The number of legs the tensor have.
    shape : tuple of int
    
    """
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









