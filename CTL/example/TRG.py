from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.contract import contractTensors
from CTL.tensor.tensor import makeTriangleTensor
import numpy as np

# Tensor Renormalization Group

# triangle lattice TRG, PRL 99, 120601 (2007)

class TriangleTRG:

    def __init__(self, alpha):
        self.alpha = alpha 
    
    def prepareInitialTensor(self):
        a = np.zeros((2, 2, 2), dtype = np.float64)
        a[(0, 0, 0)] = 1.0
        a[(0, 0, 1)] = a[(0, 1, 0)] = a[(1, 0, 0)] = self.alpha 
        self.a = makeTriangleTensor(a)
        self.b = makeTriangleTensor(a)

    def iterate(self):
        pass
    #     ta, tb = self.tensor.copyN(2)
    #     makeLink('u', 'u', ta, tb)
    #     a = contractTensors(ta, tb)
        # self.tensor = triangleContract(a)

# Given two tensors A, B
# svd on both, then contract on a square to obtain new A, B