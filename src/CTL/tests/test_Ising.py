from CTL.models.Ising import IsingSiteTensor, IsingEdgeMatrix
from CTL.tests.packedTest import PackedTest
import CTL.funcs.funcs as funcs

import numpy as np 
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