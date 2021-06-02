# make tensors for Ising model

from CTL.tensor.tensor import Tensor 
import CTL.funcs.funcs as funcs
import numpy as np 

def squareIsingTensor(beta):
    # tensor for square Ising model
    # use the simplest way to build on plaquettes, and linked with domain wall
    data = np.zeros((2, 2, 2, 2), dtype = np.float64)
    for s in range(16):
        idx = funcs.intToBitTuple(s, 4)
        localE = 0.0
        # idxSum = 0
        # if 0, then localE += beta * 0.5
        # otherwise, localE -= beta * 0.5
        # sum must be even

        for i in range(4):
            # idxSum += idx[i]
            if (idx[i] == idx[(i + 1) % 4]):
                localE -= 1.0
            else:
                localE += 1.0
        
        # if (idxSum % 2 == 0):
        data[idx] = np.exp(beta * localE)
        # else:
        #     data[idx] = 0.0

    return Tensor(labels = ['u', 'l', 'd', 'r'], data = data, degreeOfFreedom = 2)