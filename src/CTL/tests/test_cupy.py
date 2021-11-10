from CTL.tensor.tensor import Tensor 
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensornetwork.tensordict import TensorDict
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractTensorList, generateOptimalSequence, contractWithSequence

from CTL.models.Ising import squareIsingTensor, infiniteIsingExactM
from CTL.examples.impurity import ImpurityTensorNetwork
from CTL.examples.HOTRG import HOTRG

import numpy as np  
import CTL.funcs.xplib as xplib
import CTL

import CTL.funcs.funcs as funcs 
from CTL.tests.packedTest import PackedTest

def simplestExample():
    shapeA = (300, 4, 5)
    shapeB = (300, 6)
    shapeC = (4, 6, 5)
    a = Tensor(labels = ['a300', 'b4', 'c5'], data = xplib.xp.ones(shapeA))
    b = Tensor(labels = ['a300', 'd6'], data = xplib.xp.ones(shapeB))
    c = Tensor(labels = ['e4', 'd6', 'c5'], data = xplib.xp.ones(shapeC))

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

    return res

class TestCUPY(PackedTest):

    def test_cupy(self):
        # pass
        # self.assertEqual(funcs.tupleProduct((2, 3)), 6)
        try:
            import cupy as cp
        except:
            return
        CTL.setXP(cp)
        res = simplestExample()
        print('res = {}'.format(res))
        
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'cupy')