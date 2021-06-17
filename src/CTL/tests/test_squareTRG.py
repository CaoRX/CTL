# import os, sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

from CTL.tests.packedTest import PackedTest
# from CTL.tensornetwork.tensordict import TensorDict 
# from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
# from CTL.tensor.tensor import Tensor
# from CTL.tensor.tensorFactory import makeSquareTensor, makeTriangleTensor
import CTL.funcs.funcs as funcs
# from CTL.tensor.contract.contractExp import squareContractFTN, triangleContractFTN, triangleTensorTrace
from CTL.examples.TRG import SquareTRG
from CTL.models.Ising import squareIsingTensor

import numpy as np 

class TestSquareTRG(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Square TRG')

    def test_squareTRG(self):
        
        # around critical temperature

        beta = 0.4

        a = squareIsingTensor(beta = beta)
        trg = SquareTRG(a, chi = 16)
        for _ in range(20):
            trg.iterate()

        # print(trg.logZDensity())

        # deep in ferromagnetic phase

        beta = 2.0
        a = squareIsingTensor(beta = beta)
        trg2 = SquareTRG(a, chi = 16)
        for _ in range(20):
            trg2.iterate()

        errorBound = 1e-4

        self.assertTrue(trg2.errors[-1][0] < errorBound)
        FDensity = trg2.logZDensity() / beta 
        # print(FDensity)
        self.assertTrue(FDensity[-1] - 2.0 < errorBound)

        # no interaction(or infinite temperature)

        a = squareIsingTensor(beta = 0.0)
        # print(a)
        trg3 = SquareTRG(a, chi = 16)
        for _ in range(20):
            trg3.iterate()
        
        logZDensity = np.array(trg3.logZDensity())
        # print(logZDensity)
        # print(logZDensity)
        # print(np.log(2) * 0.5)
        self.assertTrue(funcs.floatEqual(logZDensity[0], np.log(2)))
        self.assertTrue(funcs.floatEqual(logZDensity[-1], np.log(2)))
