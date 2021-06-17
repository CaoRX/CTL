# import os, sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

from CTL.tests.packedTest import PackedTest
# from CTL.tensornetwork.tensordict import TensorDict 
# from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
# from CTL.tensor.tensor import Tensor
# from CTL.tensor.tensorFactory import makeSquareTensor, makeTriangleTensor
# import CTL.funcs.funcs as funcs
# from CTL.tensor.contract.contractExp import squareContractFTN, triangleContractFTN, triangleTensorTrace
from CTL.examples.HOTRG import HOTRG
# from CTL.examples.TRG import SquareTRG
from CTL.models.Ising import squareIsingTensor, infiniteIsingExactM
from CTL.examples.impurity import ImpurityTensorNetwork

import numpy as np 

class TestImpurity(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Impurity')

    def test_ImpurityEnergy(self):
        
        # around critical temperature

        beta = 1.0

        print("checking Ising energy with impurity tensor, HOTRG")
        a = squareIsingTensor(beta = beta)
        hotrg = HOTRG(a, chiH = 16)
        for _ in range(20):
            hotrg.iterate()

        eTensor = squareIsingTensor(beta = beta, obs = "E", symmetryBroken = 0)
        impurityTN = ImpurityTensorNetwork([a, eTensor], 2)
        impurityTN.setRG(hotrg) 

        for _ in range(20):
            impurityTN.iterate()
        E = impurityTN.measureObservables()
        E = [x[1] for x in E]
        # print(E)
        self.assertTrue(np.abs(-4.0 - E[-1]) < 1e-2)

    def test_ImpurityMagnetMoment(self):

        print("checking Ising magnet with impurity tensor, HOTRG, beta = 1.0")

        beta = 0.5

        symmetryBroken = 1e-5
        a = squareIsingTensor(beta = beta, symmetryBroken = symmetryBroken)
        hotrg = HOTRG(a, chiH = 16)
        for _ in range(20):
            hotrg.iterate()
        
        mTensor = squareIsingTensor(beta = beta, obs = "M", symmetryBroken = symmetryBroken)
        impurityTN = ImpurityTensorNetwork([a, mTensor], 2)
        impurityTN.setRG(hotrg) 

        for _ in range(20):
            impurityTN.iterate()
        M = impurityTN.measureObservables()
        M = [x[1] for x in M]
        exactM = infiniteIsingExactM(1.0 / beta)
        print('magnet = {}'.format(M[-1] * 0.5))
        print('exact magnet = {}'.format(exactM))

        self.assertTrue(np.abs(exactM * 2.0 - M[-1]) < 1e-5)

        print("checking Ising magnet with impurity tensor, HOTRG, beta = 0.3")

        beta = 0.3
        symmetryBroken = 1e-5
        a = squareIsingTensor(beta = beta, symmetryBroken = symmetryBroken)
        hotrg = HOTRG(a, chiH = 16)
        for _ in range(20):
            hotrg.iterate()
        
        mTensor = squareIsingTensor(beta = beta, obs = "M", symmetryBroken = symmetryBroken)
        impurityTN = ImpurityTensorNetwork([a, mTensor], 2)
        impurityTN.setRG(hotrg) 

        for _ in range(20):
            impurityTN.iterate()
        M = impurityTN.measureObservables()
        M = [x[1] for x in M]
        exactM = infiniteIsingExactM(1.0 / beta)
        print('magnet = {}'.format(M[-1] * 0.5))
        print('exact magnet = {}'.format(exactM))
        self.assertTrue(np.abs(M[-1]) < 1e-2)
        # print(M)
        # self.assertTrue(np.abs(4.0 - E[-1]) < 1e-2)
        
        # E = impurityTN.measureObservables()
        # print(E)

        # HOTRGLogZDensity = hotrg.logZDensity()

        # trg = SquareTRG(a, chi = 16)
        # for _ in range(20):
        #     trg.iterate()

        # TRGLogZDensity = trg.logZDensity()

        # errorBound = 1e-5
        # self.assertTrue(np.abs(HOTRGLogZDensity[-1] - TRGLogZDensity[-1]) < errorBound)