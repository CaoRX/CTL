# make tensors for Ising model

from CTL.tensor.tensor import Tensor
from CTL.tensor.diagonalTensor import DiagonalTensor
import CTL.funcs.funcs as funcs
import numpy as np 

def squareTensorMeasure(idx, obs = None):
    if (obs is None):
        return 1.0 
    funcs.assertInSet(obs, ['M', 'E'], 'Ising obervables')
    if (obs == 'M'):
        res = 0.0
        for x in idx:
            if (x == 0):
                res -= 0.5 
            else:
                res += 0.5
        
        return res

    if (obs == 'E'):
        res = 0.0
        for i in range(4):
            if (idx[i] == idx[(i + 1) % 4]):
                res -= 1.0
            else:
                res += 1.0
        return res
    return 1.0

def squareIsingTensor(beta, obs = None, symmetryBroken = 0.0):
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
        data[idx] = squareTensorMeasure(idx, obs) * np.exp(-beta * localE)
        data[idx] *= np.exp(symmetryBroken * squareTensorMeasure(idx, 'M'))
        # else:
        #     data[idx] = 0.0
    # print(data)

    return Tensor(labels = ['u', 'l', 'd', 'r'], data = data, degreeOfFreedom = 2)

def infiniteIsingExactM(T, V = 1.0):
    criticalT = 2.0 / np.log(1.0 + np.sqrt(2))
    if (T > criticalT):
        return 0.0
    else:
        return (1.0 - np.sinh(2 * V / T) ** (-4)) ** (0.125)

def IsingEdgeMatrix(betaJ):
    '''
    edge matrix of Ising model
    when combined with diagonal tensor at sites: tensor network of Ising model
    '''
    diag = np.sqrt(np.cosh(betaJ) * 0.5) + np.sqrt(np.sinh(betaJ) * 0.5)
    offDiag = np.sqrt(np.cosh(betaJ) * 0.5) - np.sqrt(np.sinh(betaJ) * 0.5)
    return np.array([[diag, offDiag], [offDiag, diag]])

def IsingSiteTensor(betaJ, dim = 4, labels = None):
    a = np.array([1.0, 1.0])
    a = funcs.diagonalMatrix(a, dim = dim)
    edgeMat = IsingEdgeMatrix(betaJ)
    for _ in range(dim):
        a = np.tensordot(a, edgeMat, (0, 0))
        # print(a)
    return Tensor(data = a, labels = labels)
