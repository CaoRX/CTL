import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor
from CTL.tensor.tensorFactory import makeSquareTensor, makeTriangleTensor
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contractExp import squareContractFTN, triangleContractFTN, triangleTensorTrace
from CTL.examples.HOTRG import HOTRG
from CTL.examples.TRG import SquareTRG
from CTL.models.Ising import squareIsingTensor

import numpy as np 

class TestHOTRG(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'HOTRG')

    def test_HOTRG(self):
        
        # around critical temperature

        beta = 0.4

        a = squareIsingTensor(beta = beta)
        hotrg = HOTRG(a, chiH = 16)
        for _ in range(20):
            hotrg.iterate()

        HOTRGLogZDensity = hotrg.logZDensity()

        trg = SquareTRG(a, chi = 16)
        for _ in range(20):
            trg.iterate()

        TRGLogZDensity = trg.logZDensity()

        errorBound = 1e-4
        self.assertTrue(np.abs(HOTRGLogZDensity[-1] - TRGLogZDensity[-1]) < errorBound)
        # print(hotrg.logZDensity())
        # print(hotrg.errors)
        # print(trg.aArchive)

        # deep in ferromagnetic phase

        # beta = 2.0
        # a = squareIsingTensor(beta = beta)
        # trg2 = SquareTRG(a, chi = 16)
        # for _ in range(20):
        #     trg2.iterate()

        # errorBound = 1e-4

        # self.assertTrue(trg2.errors[-1][0] < errorBound)
        # FDensity = trg2.logZDensity() / beta 
        # # print(FDensity)
        # self.assertTrue(FDensity[-1] - 2.0 < errorBound)

        # # no interaction(or infinite temperature)

        # a = squareIsingTensor(beta = 0.0)
        # # print(a)
        # trg3 = SquareTRG(a, chi = 16)
        # for _ in range(20):
        #     trg3.iterate()
        
        # logZDensity = np.array(trg3.logZDensity())
        # # print(logZDensity)
        # # print(np.log(2) * 0.5)
        # self.assertTrue(funcs.floatEqual(logZDensity[0], np.log(2)))
        # self.assertTrue(funcs.floatEqual(logZDensity[-1], np.log(2)))

# if __name__ == '__main__':
#     beta = 0.4

#     a = squareIsingTensor(beta = beta)
#     trg = HOTRG(a, chiH = 16)
#     for _ in range(20):
#         trg.iterate()
#     print(trg.logZDensity())