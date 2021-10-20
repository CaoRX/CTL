from CTL.models.Ising import IsingSiteTensor, IsingEdgeMatrix, IsingTNFromUndirectedGraph, exactZFromGraphIsing
from CTL.tests.packedTest import PackedTest
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.optimalContract import contractAndCostWithSequence, generateOptimalSequence, generateGreedySequence
from CTL.examples.MPS import contractWithMPS

import numpy as np 

from CTL.funcs.graphFuncs import squareLatticePBC, squareLatticeFBC, doubleSquareLatticeFBC, completeGraph
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

        doubleLatticeFBC = doubleSquareLatticeFBC(n = 2, m = 2, weight = 0.5) # 12 tensors
        tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)

        # seq = [(1, 8), (4, 9), (5, 11), (10, 5), (4, 5), (3, 4), (1, 3), (7, 1), (2, 1), (6, 1), (0, 1)] # calculate on-the-fly is ok
        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

        exactZ = exactZFromGraphIsing(doubleLatticeFBC)
        print('exact Z = {}'.format(exactZ))

        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        print('Z from MPS = {}'.format(ZMPS.single()))

        self.assertTrue(funcs.floatRelativeEqual(Z.single(), exactZ, eps = 1e-10))
        self.assertTrue(funcs.floatRelativeEqual(ZMPS.single(), exactZ, eps = 1e-10))

        # print(tensorNetwork)

        doubleLatticeFBC = doubleSquareLatticeFBC(n = 3, m = 3, weight = 0.5) # 24 tensors
        tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)

        seq = [(2, 15), (14, 2), (5, 2), (9, 20), (21, 9), (6, 9), (16, 6), (11, 23), (22, 11), (8, 11), (19, 8), (10, 8), (6, 8), (7, 6), (18, 6), (2, 6), (17, 2), (4, 2), (1, 2), (13, 1), (3, 1), (12, 1), (0, 1)]
        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = seq)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

        # exactZ = exactZFromGraphIsing(doubleLatticeFBC)
        # print('exact Z = {}'.format(exactZ))
        exactZ = 2694263494.5463686 # pre-calculated
        print('exact Z = {}'.format(exactZ))

        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        print('Z from MPS = {}'.format(ZMPS.single()))

        self.assertTrue(funcs.floatRelativeEqual(Z.single(), exactZ, eps = 1e-10))
        self.assertTrue(funcs.floatRelativeEqual(ZMPS.single(), exactZ, eps = 1e-10))

        doubleLatticeFBC = doubleSquareLatticeFBC(n = 3, m = 3, weight = (0.5, 1.0)) # 24 tensors
        tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)

        seq = [(2, 15), (14, 2), (5, 2), (9, 20), (21, 9), (6, 9), (16, 6), (11, 23), (22, 11), (8, 11), (19, 8), (10, 8), (6, 8), (7, 6), (18, 6), (2, 6), (17, 2), (4, 2), (1, 2), (13, 1), (3, 1), (12, 1), (0, 1)]
        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = seq)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

        # exactZ = exactZFromGraphIsing(doubleLatticeFBC)
        # print('exact Z = {}'.format(exactZ))
        # exactZ = 2694263494.5463686 # pre-calculated
        # print('exact Z = {}'.format(exactZ))

        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        print('Z from MPS = {}'.format(ZMPS.single()))

        self.assertTrue(funcs.floatRelativeEqual(Z.single(), ZMPS.single(), eps = 1e-10))
        # self.assertTrue(funcs.floatRelativeEqual(ZMPS.single(), exactZ, eps = 1e-10))

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

        latticePBC = squareLatticePBC(n = 5, m = 5, weight = 0.5)
        tensorNetwork = IsingTNFromUndirectedGraph(latticePBC)
        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        # self.assertTrue(funcs.floatEqual(ZMPS.single(), 2 ** 25, eps = 1e-5))
        print('ZMPS for J = 0.5 (5, 5) = {}'.format(ZMPS.single()))

        # exactZ = exactZFromGraphIsing(latticePBC) (10 minutes needed)
        exactZ = 274435114113.4535
        print('exact Z = {}'.format(exactZ))
        self.assertTrue(funcs.floatRelativeEqual(ZMPS.single(), exactZ, eps = 1e-5))

        print('')

    def test_fullConnectedIsing(self):
        print('begin testing full connected Ising model:')
        latticeFC = completeGraph(n = 10, weight = 0.5)
        tensorNetwork = IsingTNFromUndirectedGraph(latticeFC)

        # seq = generateOptimalSequence(tensorNetwork, typicalDim = None)
        seq = generateGreedySequence(tensorNetwork)
        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = seq)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        print('Z from MPS = {}'.format(ZMPS.single()))
        
        exactZ = exactZFromGraphIsing(latticeFC)
        print('exact Z = {}'.format(exactZ))

        self.assertTrue(funcs.floatEqual(Z.single(), exactZ, eps = 1e-3))
        self.assertTrue(funcs.floatEqual(ZMPS.single(), exactZ, eps = 1e-3))

        # latticeFC = completeGraph(n = 15, weight = 0.5)
        # tensorNetwork = IsingTNFromUndirectedGraph(latticeFC)

        # # seq = generateGreedySequence(tensorNetwork)
        # # Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = seq)

        # ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        # print('Z from MPS = {}'.format(ZMPS.single()))
        
        # exactZ = exactZFromGraphIsing(latticeFC)
        # print('exact Z = {}'.format(exactZ))

        print('')


