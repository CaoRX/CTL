from CTL.models.Ising import IsingSiteTensor, IsingEdgeMatrix, IsingTNFromUndirectedGraph, exactZFromGraphIsing
from CTL.tests.packedTest import PackedTest
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.optimalContract import contractAndCostWithSequence, generateOptimalSequence
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
        
        latticePBC = squareLatticePBC(n = 4, m = 4, weight = 0.5)
        # for (4, 4) PBC case: the cost is low(13328 for full contraction), but the CPU time for calculating the optimal sequence is about one minute, so we test here with pre-calculated sequence
        tensorNetwork = IsingTNFromUndirectedGraph(latticePBC)

        # seq = generateOptimalSequence(tensorNetwork, typicalDim = None)
        seq = [(2, 3), (5, 6), (4, 7), (5, 4), (8, 12), (11, 15), (8, 11), (9, 13), (10, 14), (9, 10), (8, 9), (4, 8), (2, 4), (1, 2), (0, 1)] # pre-calculated
        # print('optimal seq = {}'.format(seq))
        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = seq)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        print('Z from MPS = {}'.format(ZMPS.single()))
        
        exactZ = exactZFromGraphIsing(latticePBC)
        print('exact Z = {}'.format(exactZ))

        self.assertTrue(funcs.floatEqual(Z.single(), exactZ, eps = 1e-4))
        self.assertTrue(funcs.floatEqual(ZMPS.single(), exactZ, eps = 1e-4))

        latticeFBC = squareLatticeFBC(n = 4, m = 4, weight = 0.5)
        tensorNetwork = IsingTNFromUndirectedGraph(latticeFBC)

        # seq = generateOptimalSequence(tensorNetwork, typicalDim = None)
        seq = [(3, 7), (2, 3), (6, 2), (12, 13), (8, 12), (9, 8), (14, 15), (11, 14), (10, 11), (8, 10), (2, 8), (5, 2), (4, 2), (1, 2), (0, 1)] # pre-calculated
        # print('optimal seq = {}'.format(seq))
        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = seq)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

        # ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16, seq = seq)
        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        print('Z from MPS = {}'.format(ZMPS.single()))
        
        exactZ = exactZFromGraphIsing(latticeFBC)
        print('exact Z = {}'.format(exactZ))

        self.assertTrue(funcs.floatEqual(Z.single(), exactZ, eps = 1e-6))
        self.assertTrue(funcs.floatEqual(ZMPS.single(), exactZ, eps = 1e-6))

    def test_largerIsing(self):
        print('begin testing larger Ising model')

        latticePBC = squareLatticePBC(n = 5, m = 5, weight = 0.0)
        tensorNetwork = IsingTNFromUndirectedGraph(latticePBC)
        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        self.assertTrue(funcs.floatEqual(ZMPS.single(), 2 ** 25, eps = 1e-5))
        print('ZMPS for (5, 5) = {}'.format(ZMPS.single()))

        latticePBC = squareLatticePBC(n = 6, m = 6, weight = 0.0)
        tensorNetwork = IsingTNFromUndirectedGraph(latticePBC)
        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        # self.assertTrue(funcs.floatEqual(ZMPS.single(), 2 ** 25, eps = 1e-5))
        print('ZMPS for (6, 6) = {}'.format(ZMPS.single()))
        self.assertTrue(funcs.floatEqual(ZMPS.single(), 2 ** 36, eps = 1e-2))


