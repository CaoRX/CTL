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
from CTL.tensor.contract.optimalContract import contractTensorList, generateOptimalSequence, contractWithSequence

from CTL.models.Ising import squareIsingTensor, infiniteIsingExactM
from CTL.examples.impurity import ImpurityTensorNetwork
from CTL.examples.HOTRG import HOTRG

import numpy as np  

def simplestExample():
    shapeA = (300, 4, 5)
    shapeB = (300, 6)
    shapeC = (4, 6, 5)
    a = Tensor(labels = ['a300', 'b4', 'c5'], data = np.ones(shapeA))
    b = Tensor(labels = ['a300', 'd6'], data = np.ones(shapeB))
    c = Tensor(labels = ['e4', 'd6', 'c5'], data = np.ones(shapeC))

    # create tensors with labels

    makeLink('a300', 'a300', a, b)
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
    res = contractWithSequence([a, b, c], seq = optimalSeq)
    print(res)

    # if you want to save time / space by contract in place(note that after this you cannot contract them again, since their bonds between have been broken):
    res = contractWithSequence([a, b, c], seq = optimalSeq, inplace = True)
    print(res)
    print('')

    # for reusable inplace contraction(which is our goal), refer to the use of CTL.tensornetwork.tensornetwork.FiniteTensorNetwork

def HOTRGImpurityExample(beta = 0.5):
    print('test magnet for Ising model, beta = {}'.format(beta))
    # beta = 0.6

    symmetryBroken = 1e-5
    a = squareIsingTensor(beta = beta, symmetryBroken = symmetryBroken)
    hotrg = HOTRG(a, chiH = 16)
    for _ in range(20):
        hotrg.iterate()
    
    mTensor = squareIsingTensor(beta = beta, obs = "M", symmetryBroken = symmetryBroken)
    impurityTN = ImpurityTensorNetwork([a, mTensor], 2)
    impurityTN.setRG(hotrg) 

    for _ in range(20):
        impurityTN.iterate()
    M = impurityTN.measureObservables()
    M = [x[1] for x in M]
    exactM = infiniteIsingExactM(1.0 / beta)
    print('magnet = {}'.format(M[-1] * 0.5))
    print('exact magnet = {}'.format(exactM))

if __name__ == '__main__':
    simplestExample()
    HOTRGImpurityExample(beta = 0.6)