from tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor, makeSquareTensor, makeTriangleTensor
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contractExp import squareContractFTN, triangleContractFTN

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
