# calculation of MPS
# given a tensor, return the MPS by taking each bond as an MPS tensor

import CTL.funcs.funcs as funcs

class FreeBoundaryMPS:

    '''
    MPS:
    Maintain a list of tensors, the first & last tensors are 2-dimensional, others are 3-dimensional
    The outer bonds are just the legs of the input tensor, and the internal bonds are with bond dimension at max chi
    Default to be canonical form? Support tensor swap?
    '''

    def checkMPSProperty(self, tensorList):
        return True

    def __init__(self, tensorList):
        if (not self.checkMPSProperty(tensorList)):
            raise ValueError(funcs.errorMessage("tensorList {} cannot be considered as an MPS".format(tensorList), location = "FreeBoundaryMPS.__init__"))
        pass

    def canonicalize(self):
        pass 

    def swap(self, tensorA, tensorB):
        # tensorA and tensorB are tensors in tensorList
        pass 