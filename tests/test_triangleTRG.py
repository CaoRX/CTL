from tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor
from CTL.tensor.tensorFactory import makeSquareTensor, makeTriangleTensor
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contractExp import squareContractFTN, triangleContractFTN, triangleTensorTrace
from CTL.examples.TRG import TriangleTRG

import numpy as np 

class TestTriangleTRG(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Triangle TRG')

    def test_triangleTRG(self):

        errorUpBound = 1e-4

        trg = TriangleTRG(alpha = 0.5, chi = 16)
        for _ in range(20):
            trg.iterate()

        # print(trg.aNorms, trg.bNorms)
        # print('error = {}'.format(trg.errors))

        trg2 = TriangleTRG(alpha = 0.6, chi = 16)
        for _ in range(20):
            trg2.iterate()

        self.assertTrue(funcs.floatEqual(trg2.aArchive[-1].a[(0, 0, 0)], -0.96118630, eps = 1e-7))

        # print(trg.aNorms, trg.bNorms)
        # print('error = {}'.format(trg2.errors))
        # print(trg2.aArchive)
        # for tensor in trg2.aArchive:
        #     print(tensor.a[(0, 0, 0)])

        trg3 = TriangleTRG(alpha = 0.6, chi = 20)
        for _ in range(20):
            trg3.iterate()

        self.assertTrue(funcs.floatEqual(trg3.aArchive[-1].a[(0, 0, 0)], -0.96118630, eps = 1e-7))

        self.assertTrue(trg.errors[-1] < errorUpBound)
        self.assertTrue(trg2.errors[-1] < errorUpBound)
        self.assertTrue(trg3.errors[-1] < errorUpBound)

        # print('trace = {}'.format(triangleTensorTrace(trg3.a, trg3.b).single()))

        # print(trg.aNorms, trg.bNorms)
        # print('error = {}'.format(trg3.errors))
        # print(trg3.aArchive)
        # for tensor in trg3.aArchive:
        #     print(tensor.a[(0, 0, 0)])

    def test_triangleTRGResults(self):
        # check the results of TRG to compare with PRL 99, 120601 (2007)
        trg = TriangleTRG(alpha = 0.5, chi = 16)
        for _ in range(20):
            trg.iterate()

        # print('logZ = {}'.format(trg.logZDensity()))

        trg2 = TriangleTRG(alpha = 1.5, chi = 16)
        for _ in range(20):
            trg2.iterate()

        # print('logZ = {}'.format(trg2.logZDensity()))

        pass