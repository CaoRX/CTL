# import os, sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

from tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor
from CTL.tensor.tensorFactory import makeSquareTensor, makeTriangleTensor
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contractExp import squareContractFTN, triangleContractFTN, triangleTensorTrace
from CTL.examples.TRG import SquareTRG
from CTL.models.Ising import squareIsingTensor

import numpy as np 

class TestSquareTRG(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Square TRG')

    def test_squareTRG(self):

        beta = 0.4

        a = squareIsingTensor(beta = beta)
        trg = SquareTRG(a, chi = 16)
        for _ in range(20):
            trg.iterate()

        # print(trg.logZDensity() / beta) # Free Energy per site
        # print(trg.errors)

        beta = 2.0
        a = squareIsingTensor(beta = beta)
        trg2 = SquareTRG(a, chi = 16)
        for _ in range(20):
            trg2.iterate()

        errorBound = 1e-4

        self.assertTrue(trg2.errors[-1][0] < errorBound)
        FDensity = trg2.logZDensity() / beta 
        self.assertTrue(FDensity[-1] - 2.0 < errorBound)

        # print(trg2.logZDensity() / beta)
        # print(trg2.errors)

        # print([a.a[(0, 0, 0, 0)] for a in trg.aArchive])
        # print(trg.errors)

        # a = squareIsingTensor(beta = 0.4)
        # trg2 = SquareTRG(a, chi = 20)
        # for _ in range(20):
        #     trg2.iterate()

        # print([a.a[(0, 0, 0, 0)] for a in trg2.aArchive])
        # print(trg2.errors)

        a = squareIsingTensor(beta = 0.0)
        trg3 = SquareTRG(a, chi = 16)
        for _ in range(20):
            trg3.iterate()
        
        logZDensity = np.array(trg3.logZDensity())
        self.assertTrue(funcs.floatEqual(logZDensity[0], np.log(2)))
        self.assertTrue(funcs.floatEqual(logZDensity[-1], np.log(2)))

        
        # dof = np.array([x.degreeOfFreedom for x in trg3.aArchive])
        # print(logZDensity)
        # logZDensity = logZDensity + np.log(2) / dof
        # print(logZDensity)
        # print(np.exp(logZDensity))
        # Z = np.exp(logZDensity * dof)
        # print(Z, dof)
        # print([a.a[(0, 0, 0, 0)] for a in trg3.aArchive])
        # print(trg3.errors)

        # errorUpBound = 1e-4

        # trg = TriangleTRG(alpha = 0.5, chi = 16)
        # for _ in range(20):
        #     trg.iterate()

        # # print(trg.aNorms, trg.bNorms)
        # # print('error = {}'.format(trg.errors))

        # trg2 = TriangleTRG(alpha = 0.6, chi = 16)
        # for _ in range(20):
        #     trg2.iterate()

        # self.assertTrue(funcs.floatEqual(trg2.aArchive[-1].a[(0, 0, 0)], -0.96118630, eps = 1e-7))

        # # print(trg.aNorms, trg.bNorms)
        # # print('error = {}'.format(trg2.errors))
        # # print(trg2.aArchive)
        # # for tensor in trg2.aArchive:
        # #     print(tensor.a[(0, 0, 0)])

        # trg3 = TriangleTRG(alpha = 0.6, chi = 20)
        # for _ in range(20):
        #     trg3.iterate()

        # self.assertTrue(funcs.floatEqual(trg3.aArchive[-1].a[(0, 0, 0)], -0.96118630, eps = 1e-7))

        # self.assertTrue(trg.errors[-1] < errorUpBound)
        # self.assertTrue(trg2.errors[-1] < errorUpBound)
        # self.assertTrue(trg3.errors[-1] < errorUpBound)

        # print('trace = {}'.format(triangleTensorTrace(trg3.a, trg3.b).single()))

        # print(trg.aNorms, trg.bNorms)
        # print('error = {}'.format(trg3.errors))
        # print(trg3.aArchive)
        # for tensor in trg3.aArchive:
        #     print(tensor.a[(0, 0, 0)])

# if __name__ == '__main__':
#     a = squareIsingTensor(beta = 0.4)
#     trg = SquareTRG(a, chi = 16)
#     for _ in range(20):
#         trg.iterate()

#     print(trg.aArchive)