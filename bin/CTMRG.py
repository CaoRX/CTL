import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'src'))

from CTL.models.Ising import plaquetteIsingTensor, IsingTNFromUndirectedGraph, exactZFromGraphIsing
from CTL.examples.CTMRG import CTMRG

from CTL.funcs.graphFuncs import doubleSquareLatticeFBC
from CTL.tensor.contract.optimalContract import contractAndCostWithSequence
from CTL.examples.MPS import contractWithMPS

import CTL.funcs.funcs as funcs

def squareIsingTest(L = 4):
    print('squareIsingTest(L = {}):'.format(L))

    weight = 0.5

    a = plaquetteIsingTensor(weight = weight, diamondForm = True)
    ctmrg = CTMRG(a, chi = 16)
    ZCTMRG = ctmrg.getZ(L = L)
    print('CTMRG Z = {}'.format(ZCTMRG))

    latticeFBC = doubleSquareLatticeFBC(n = L, m = L, weight = weight)
    tensorNetwork = IsingTNFromUndirectedGraph(latticeFBC)

    if (L <= 6):
        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, greedy = True)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

    ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
    print('Z from MPS = {}'.format(ZMPS.single()))
    
    if (L <= 3):
        exactZ = exactZFromGraphIsing(latticeFBC)
        print('exact Z = {}'.format(exactZ))

if __name__ == '__main__':
    squareIsingTest(L = 4)
    squareIsingTest(L = 6)
