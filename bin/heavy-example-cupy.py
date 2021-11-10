# a simple example about how to use the library

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'src'))

# import the functions and Classes we will use
# the import now is a little troublesome to find where the function you want is
# it is planned to be improved, maybe like numpy structure(so all functions can be used with CTL.xx?)
from CTL.tensor.tensor import Tensor 
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensornetwork.tensordict import TensorDict
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractTensorList, generateOptimalSequence, contractWithSequence, contractAndCostWithSequence

from CTL.models.Ising import squareIsingTensor, infiniteIsingExactM
from CTL.examples.impurity import ImpurityTensorNetwork
from CTL.examples.HOTRG import HOTRG

import numpy as np  
import cupy as cp
import CTL

def example():
    shapeA = (300, 400, 50)
    shapeB = (300, 60)
    shapeC = (400, 60, 50)
    a = Tensor(labels = ['a3', 'b4', 'c5'], data = cp.ones(shapeA))
    b = Tensor(labels = ['a3', 'd6'], data = cp.ones(shapeB))
    c = Tensor(labels = ['e4', 'd6', 'c5'], data = cp.ones(shapeC))

    # create tensors with labels

    makeLink('a3', 'a3', a, b)
    makeLink('c5', 'c5', a, c)
    makeLink('d6', 'd6', b, c)

    # make links via labels
    # note that labels can also be made between the "Leg" objects to avoid reused leg names, but here for simplicity from leg names

    # now we have a tensor network, we can generate the optimal sequence of this tensor list
    optimalSeq = generateOptimalSequence([a, b, c])
    print('optimal contraction sequence = {}'.format(optimalSeq))

    # if we do not have any knowledge in prior, we can contract the tensor list like
    res = contractTensorList([a, b, c])
    print(res)

    # if you already have a good sequence to use
    res, cost = contractAndCostWithSequence([a, b, c], seq = optimalSeq)
    print(res)
    print('contraction cost = {}'.format(cost))

    # if you want to save time / space by contract in place(note that after this you cannot contract them again, since their bonds between have been broken):
    res = contractWithSequence([a, b, c], seq = optimalSeq, inplace = True)
    print(res)
    print('')

    # for reusable inplace contraction(which is our goal), refer to the use of CTL.tensornetwork.tensornetwork.FiniteTensorNetwork

if __name__ == '__main__':
    CTL.setXP(cp)
    for _ in range(10):
        example()