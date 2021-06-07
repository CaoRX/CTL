from tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
# from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
# from CTL.tensor.tensor import Tensor
from CTL.tensor.tensorFactory import makeSquareTensor, makeTriangleTensor, makeSquareOutTensor
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contractExp import squareContractFTN, triangleContractFTN, squareContractOutFTN

import numpy as np 

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

