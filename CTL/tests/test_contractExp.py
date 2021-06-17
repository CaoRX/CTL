from tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
# from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor
from CTL.tensor.tensorFactory import makeSquareTensor, makeTriangleTensor, makeSquareOutTensor
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contractExp import squareContractFTN, triangleContractFTN, squareContractOutFTN, EvenblyTNRQEnvFTN

import numpy as np 
# from ncon import ncon

class TestContractExamples(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Contract Examples')

    def test_triangleContractFTN(self):
        FTN = triangleContractFTN()
        a = makeTriangleTensor(np.ones((3, 3, 3)))
        tensorDict = TensorDict({'u': a, 'l': a, 'r': a})

        res = FTN.contract(tensorDict)
        self.assertTrue(funcs.compareLists(res.labels, ['1', '2', '3']))
        res.reArrange(['1', '2', '3'])
        self.assertTupleEqual(res.shape, (3, 3, 3))
        self.assertTrue(FTN.locked)

    def test_squareContractFTN(self):

        FTN = squareContractFTN()
        a = makeSquareTensor(np.ones((3, 4, 3, 4)))
        tensorDict = TensorDict({'ul': a, 'ur': a, 'dl': a, 'dr': a})

        res = FTN.contract(tensorDict)
        # print(res.labels
        self.assertTrue(funcs.compareLists(res.labels, ['u', 'l', 'd', 'r']))

        res.reArrange(['u', 'l', 'd', 'r'])
        self.assertTupleEqual(res.shape, (9, 16, 9, 16))

    def test_squareContractOutFTN(self):
        FTN = squareContractOutFTN()

        tensorNames = ['ul', 'ur', 'dl', 'dr']
        tensorDict = TensorDict()
        # print('keys = {}'.format(tensorDict.tensors.keys()))
        for tensorName in tensorNames:
            tensorDict.setTensor(tensorName, makeSquareOutTensor(np.ones((3, 4, 6)), loc = tensorName))
        # print('keys = {}'.format(tensorDict.tensors.keys()))

        res = FTN.contract(tensorDict)
        self.assertTrue(funcs.compareLists(res.labels, ['u', 'd', 'l', 'r']))
        self.assertTupleEqual(res.shape, (6, 6, 6, 6))
        self.assertEqual(int(res.a[(0, 0, 0, 0)]), 144)
        # print(res.a)
        # print(res)

    # this test only works if ncon has been installed, although not necessary in CTL

    def test_qEnv(self):

        try:
            from ncon import ncon
        except:
            print("ncon is not installed. Skip the qEnv test.")
            return 
        print('checking finite tensor network with qEnv network from Evenbly TNR')
        chiHI = 5
        chiVI = 6
        tensorArray = []
        for _ in range(8):
            tensorArray.append(np.random.rand(chiHI, chiVI, chiHI, chiVI))
        qEnv = ncon(tensorArray,[[-1,-2,11,12],[7,8,11,9],[5,12,1,2],[5,9,3,4],
        [-3,-4,13,14],[7,8,13,10],[6,14,1,2],[6,10,3,4]]).reshape(chiHI*chiVI,chiHI*chiVI)
        # by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 29/1/2019

        FTN = EvenblyTNRQEnvFTN()
        tensorDict = TensorDict()
        tensorNames = ['uul', 'uur', 'udl', 'udr', 'ddl', 'ddr', 'dul', 'dur']
        for tensor, name in zip(tensorArray, tensorNames):
            t = Tensor(data = tensor, labels = ['l', 'u', 'r', 'd'])
            tensorDict.setTensor(name, t)

        res = FTN.contract(tensorDict)
        resMat = res.toMatrix(rows = ['1'], cols = ['2'])
        error = np.linalg.norm(resMat - qEnv) / np.linalg.norm(resMat)
        print('qenv error = {}'.format(error))
        self.assertTrue(error < 1e-10)

        

