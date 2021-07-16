from CTL.models.Ising import IsingSiteTensor, IsingEdgeMatrix, IsingTNFromUndirectedGraph, exactZFromGraphIsing
from CTL.tests.packedTest import PackedTest
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.optimalContract import contractAndCostWithSequence
from CTL.examples.MPS import contractWithMPS

import numpy as np 

from CTL.funcs.graphFuncs import squareLatticePBC, squareLatticeFBC
# from ncon import ncon

class TestIsing(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Ising')

    def test_Ising(self):
        betaJ = 1.0
        a = IsingSiteTensor(betaJ = betaJ, dim = 4, labels = ['u', 'd', 'l', 'r'])
        edgeMat = IsingEdgeMatrix(betaJ = betaJ)
        self.assertTrue(funcs.floatEqual(a.a[(0, 0, 0, 0)], edgeMat[0, 0] ** 4 + edgeMat[0, 1] ** 4))
        self.assertTrue(funcs.compareLists(['u', 'd', 'l', 'r'], a.labels))

    def test_IsingFromGraph(self):
        latticePBC = squareLatticePBC(n = 3, m = 3, weight = 0.5)
        tensorNetwork = IsingTNFromUndirectedGraph(latticePBC)

        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork)
        # print('Z = {}, cost = {}'.format(Z.single(), cost))

        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        # print('Z from MPS = {}'.format(ZMPS.single()))
        
        exactZ = exactZFromGraphIsing(latticePBC)
        # print('exact Z = {}'.format(exactZ))

        self.assertTrue(funcs.floatEqual(Z.single(), exactZ, eps = 1e-6))
        self.assertTrue(funcs.floatEqual(ZMPS.single(), exactZ, eps = 1e-6))

        latticeFBC = squareLatticeFBC(n = 4, m = 4, weight = 0.5)
        tensorNetwork = IsingTNFromUndirectedGraph(latticeFBC)

        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork)
        # print('Z = {}, cost = {}'.format(Z.single(), cost))

        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        # print('Z from MPS = {}'.format(ZMPS.single()))
        
        exactZ = exactZFromGraphIsing(latticeFBC)
        # print('exact Z = {}'.format(exactZ))

        self.assertTrue(funcs.floatEqual(Z.single(), exactZ, eps = 1e-6))
        self.assertTrue(funcs.floatEqual(ZMPS.single(), exactZ, eps = 1e-6))


