from CTL.tests.packedTest import PackedTest

from CTL.models.Ising import plaquetteIsingTensor, IsingTNFromUndirectedGraph
from CTL.examples.CTMRG import CTMRG

from CTL.funcs.graphFuncs import doubleSquareLatticeFBC
from CTL.tensor.contract.optimalContract import contractAndCostWithSequence
from CTL.examples.MPS import contractWithMPS

import CTL.funcs.funcs as funcs

class TestCTMRG(PackedTest):
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'CTMRG')

    def test_exactCTMRG(self):
        # test case for non-interacting Ising model
        weight = 0.0
        doubleLatticeFBC = doubleSquareLatticeFBC(n = 3, m = 3, weight = weight) # 24 tensors
        tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)

        seq = [(2, 15), (14, 2), (5, 2), (9, 20), (21, 9), (6, 9), (16, 6), (11, 23), (22, 11), (8, 11), (19, 8), (10, 8), (6, 8), (7, 6), (18, 6), (2, 6), (17, 2), (4, 2), (1, 2), (13, 1), (3, 1), (12, 1), (0, 1)]
        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = seq)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

        # exactZ = 2694263494.5463686 # pre-calculated
        # print('exact Z = {}'.format(exactZ))

        # ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        # print('Z from MPS = {}'.format(ZMPS.single()))

        a = plaquetteIsingTensor(weight = weight, diamondForm = True)
        
        ctmrg = CTMRG(a, chi = 16)
        # for i in range(1, 5):
        #     print('CTMRG Z(L = {}) = {}'.format(i, ctmrg.getZ(L = i)))
        # with self.assertWarns(RuntimeWarning):
        ZCTMRG = ctmrg.getZ(L = 3)
        print('CTMRG Z = {}'.format(ZCTMRG))
        self.assertTrue(funcs.floatRelativeEqual(ZCTMRG, Z.single(), eps = 1e-10))
        
        # test case for Ising model
        weight = 0.5

        for nn in range(1, 3):
            doubleLatticeFBC = doubleSquareLatticeFBC(n = nn, m = nn, weight = weight) # 24 tensors
            tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)
            Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork)
            print('Z for L = {} is {}'.format(nn, Z.single()))

        doubleLatticeFBC = doubleSquareLatticeFBC(n = 3, m = 3, weight = weight) # 24 tensors
        tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)

        seq = [(2, 15), (14, 2), (5, 2), (9, 20), (21, 9), (6, 9), (16, 6), (11, 23), (22, 11), (8, 11), (19, 8), (10, 8), (6, 8), (7, 6), (18, 6), (2, 6), (17, 2), (4, 2), (1, 2), (13, 1), (3, 1), (12, 1), (0, 1)]
        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = seq)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

        # exactZ = 2694263494.5463686 # pre-calculated
        # print('exact Z = {}'.format(exactZ))

        # ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        # print('Z from MPS = {}'.format(ZMPS.single()))

        a = plaquetteIsingTensor(weight = weight, diamondForm = True)
        # for i in range(1, 5):
        #     print('CTMRG Z(L = {}) = {}'.format(i, ctmrg.getSingleZ(L = i)))
        
        ctmrg = CTMRG(a, chi = 16)
        # for i in range(1, 5):
        #     print('CTMRG Z(L = {}) = {}'.format(i, ctmrg.getZ(L = i)))
        # with self.assertWarns(RuntimeWarning):
        ZCTMRG = ctmrg.getZ(L = 3)
        print('CTMRG Z = {}'.format(ZCTMRG))
        self.assertTrue(funcs.floatRelativeEqual(ZCTMRG, Z.single(), eps = 1e-10))

        weight = 0.5
        doubleLatticeFBC = doubleSquareLatticeFBC(n = 5, m = 5, weight = weight) # 24 tensors
        tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)

        Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = None, greedy = True)
        print('Z = {}, cost = {}'.format(Z.single(), cost))

        # ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        # print('Z from MPS = {}'.format(ZMPS.single()))

        a = plaquetteIsingTensor(weight = weight, diamondForm = True)
        
        ctmrg = CTMRG(a, chi = 16)
        ZCTMRG = ctmrg.getZ(L = 5)
        print('CTMRG Z = {}'.format(ZCTMRG))
        self.assertTrue(funcs.floatRelativeEqual(ZCTMRG, Z.single(), eps = 1e-10))

        weight = 0.5
        doubleLatticeFBC = doubleSquareLatticeFBC(n = 7, m = 7, weight = weight) # 24 tensors
        tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)

        # Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = None, greedy = True)
        # print('Z = {}, cost = {}'.format(Z.single(), cost))

        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        print('Z from MPS = {}'.format(ZMPS.single()))

        a = plaquetteIsingTensor(weight = weight, diamondForm = True)
        
        ctmrg = CTMRG(a, chi = 16)
        ZCTMRG = ctmrg.getZ(L = 7)
        print('CTMRG Z = {}'.format(ZCTMRG))
        self.assertTrue(funcs.floatRelativeEqual(ZCTMRG, ZMPS.single(), eps = 1e-10))

        weight = 0.7
        doubleLatticeFBC = doubleSquareLatticeFBC(n = 6, m = 6, weight = weight) # 24 tensors
        tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)

        # Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = None, greedy = True)
        # print('Z = {}, cost = {}'.format(Z.single(), cost))

        ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        print('Z from MPS = {}'.format(ZMPS.single()))

        a = plaquetteIsingTensor(weight = weight, diamondForm = True)
        
        ctmrg = CTMRG(a, chi = 16)
        ZCTMRG = ctmrg.getZ(L = 6)
        print('CTMRG Z = {}'.format(ZCTMRG))
        self.assertTrue(funcs.floatRelativeEqual(ZCTMRG, ZMPS.single(), eps = 1e-10))

        # test case for J1-J2 Ising model(not work for current CTMRG assuming symmetry)
        # weight = (0.3, 0.4)

        # doubleLatticeFBC = doubleSquareLatticeFBC(n = 3, m = 3, weight = weight) # 24 tensors
        # tensorNetwork = IsingTNFromUndirectedGraph(doubleLatticeFBC)

        # seq = [(2, 15), (14, 2), (5, 2), (9, 20), (21, 9), (6, 9), (16, 6), (11, 23), (22, 11), (8, 11), (19, 8), (10, 8), (6, 8), (7, 6), (18, 6), (2, 6), (17, 2), (4, 2), (1, 2), (13, 1), (3, 1), (12, 1), (0, 1)]
        # Z, cost = contractAndCostWithSequence(tensorList = tensorNetwork, seq = seq)
        # print('Z = {}, cost = {}'.format(Z.single(), cost))

        # # exactZ = 2694263494.5463686 # pre-calculated
        # # print('exact Z = {}'.format(exactZ))

        # # ZMPS = contractWithMPS(tensorList = tensorNetwork, chi = 16)
        # # print('Z from MPS = {}'.format(ZMPS.single()))

        # a = plaquetteIsingTensor(weight = weight, diamondForm = True)
        # # for i in range(1, 5):
        # #     print('CTMRG Z(L = {}) = {}'.format(i, ctmrg.getSingleZ(L = i)))
        
        # ctmrg = CTMRG(a, chi = 16)
        # # with self.assertWarns(RuntimeWarning):
        # ZCTMRG = ctmrg.getZ(L = 3)
        # print('CTMRG Z = {}'.format(ZCTMRG))
        # self.assertTrue(funcs.floatRelativeEqual(ZCTMRG, Z.single(), eps = 1e-10))
