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
        
        # test case for Ising model
        weight = 0.5
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
        with self.assertWarns(RuntimeWarning):
            ZCTMRG = ctmrg.getSingleZ(L = 3)
        print('CTMRG Z = {}'.format(ZCTMRG))
        self.assertTrue(funcs.floatRelativeEqual(ZCTMRG, Z.single(), eps = 1e-10))

        # test case for J1-J2 Ising model
        weight = (0.3, 0.4)

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
        with self.assertWarns(RuntimeWarning):
            ZCTMRG = ctmrg.getSingleZ(L = 3)
        print('CTMRG Z = {}'.format(ZCTMRG))
        self.assertTrue(funcs.floatRelativeEqual(ZCTMRG, Z.single(), eps = 1e-10))
