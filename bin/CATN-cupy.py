# "Contracting Arbitrary Tensor Networks"
# PhysRevLett. 125. 060503.

# the framework of tensor contraction:
# 1. decompose each tensor into an MPS, each bond corresponding to a tensor
# 2. contract MPSes by first moving the bond to one end, and then contract
# 3. In the new tensor network, double edges should be merged by first move them to neighbors, then merge
# 4. iterate until one scalar remains

# the functionality we want:
# 1. decompose a tensor to MPS, in canonical form

# "A Practical Introduction to Tensor Networks: MPS and PEPS"

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'src'))

from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractAndCostWithSequence
from CTL.examples.MPS import contractWithMPS

from CTL.models.Ising import IsingSiteTensor, IsingEdgeMatrix, IsingTNFromUndirectedGraph, exactZFromGraphIsing
from CTL.funcs.graphFuncs import squareLatticeFBC, squareLatticeFBC

import numpy as np
import cupy as cp
import CTL

CTL.setXP(cp)

def contractHandmadeTN():
    print('contractHandmadeTN():')
    a = Tensor(shape = (3, 5, 7), labels = ['a3', 'a5', 'a7'])
    b = Tensor(shape = (2, 4, 5), labels = ['b2', 'b4', 'b5'])
    c = Tensor(shape = (2, 7, 7, 7), labels = ['c2', 'c71', 'c72', 'c73'])
    d = Tensor(shape = (7, 7, 3, 4), labels = ['d71', 'd72', 'd3', 'd4'])
    e = Tensor(shape = (3, 3, 5), labels = ['e31', 'e32', 'e5'])
    f = Tensor(shape = (2, 2, 5), labels = ['f21', 'f22', 'f5'])
    g = Tensor(shape = (4, 4, 3, 3), labels = ['g41', 'g42', 'g31', 'g32'])
    makeLink('a3', 'e31', a, e)
    makeLink('a5', 'b5', a, b)
    makeLink('a7', 'c72', a, c) 

    makeLink('b2', 'f21', b, f)
    makeLink('b4', 'g41', b, g)
    
    makeLink('c2', 'f22', c, f)
    makeLink('c71', 'd72', c, d)
    makeLink('c73', 'd71', c, d) 

    makeLink('d3', 'g31', d, g)
    makeLink('d4', 'g42', d, g)
    
    makeLink('e5', 'f5', e, f)
    makeLink('e32', 'g32', e, g)

    tensors = [a, b, d, c, g, f, e]

    res, _ = contractAndCostWithSequence(tensors)
    print('res from direct contraction = {}'.format(res.single()))

    mpsRes = contractWithMPS(tensors, chi = 32)
    print('res from mps = {}'.format(mpsRes.single()))
    print('')

def squareIsingTest():
    print('squareIsingTest():')
    latticeFBC = squareLatticeFBC(n = 4, m = 4, weight = 0.5)
    tensorNetwork = IsingTNFromUndirectedGraph(latticeFBC)

    Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork)
    print('Z = {}, cost = {}'.format(Z.single(), cost))

    ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
    print('Z from MPS = {}'.format(ZMPS.single()))
    
    exactZ = exactZFromGraphIsing(latticeFBC)
    print('exact Z = {}'.format(exactZ))

if __name__ == '__main__':
    contractHandmadeTN()
    squareIsingTest()