import unittest 
from CTL.tests.packedTest import PackedTest

import CTL.funcs.funcs as funcs
import numpy as np

class TestFuncs(PackedTest):
    def test_sumOnAxis(self):
        a = np.random.rand(3, 4, 5)
        b = np.random.rand(4)

        funcsRes = funcs.sumOnAxis(a, 1, weights = b)
        ans = np.einsum('ijk,j->ik', a, b)

        self.assertTrue(funcs.matDiffError(funcsRes, ans) < 1e-10)
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Funcs')