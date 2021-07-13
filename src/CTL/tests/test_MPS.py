import unittest 
from CTL.tensor.tensor import Tensor
import numpy as np
from CTL.tests.packedTest import PackedTest
import CTL.funcs.funcs as funcs
from CTL.examples.MPS import FreeBoundaryMPS
from CTL.tensor.contract.link import makeLink

class TestMPS(PackedTest):
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'MPS')
    
    def test_MPS(self):
        
        tensorL = Tensor(data = np.random.random_sample((3, 3)), labels = ['o', 'internal'])
        tensor1 = Tensor(data = np.random.random_sample((3, 5, 4)), labels = ['itl', 'oo', 'itr'])
        tensor2 = Tensor(data = np.random.random_sample((4, 2, 4)), labels = ['itl', 'oo', 'itr'])
        tensor3 = Tensor(data = np.random.random_sample((4, 3, 2)), labels = ['itl', 'oo', 'itr'])
        tensorR = Tensor(data = np.random.random_sample((2, 5)), labels = ['internal', 'o'])

        makeLink('internal', 'itl', tensorL, tensor1)
        makeLink('itr', 'itl', tensor1, tensor2)
        makeLink('itr', 'itl', tensor2, tensor3)
        makeLink('itr', 'internal', tensor3, tensorR)

        tensors = [tensorL, tensor1, tensor2, tensor3, tensorR]

        mps = FreeBoundaryMPS(tensorList = tensors, chi = 16)
        self.assertEqual(mps.chi, 16)
        self.assertEqual(mps.n, 5)
        # print(mps)

        self.assertTrue(mps.checkMPSProperty(mps.tensors))

        mps.canonicalize(direct = 0)
        # print(mps)
        mps.canonicalize(direct = 0)
        # print(mps)

        self.assertTrue(mps.checkCanonical(direct = 0))
        self.assertFalse(mps.checkCanonical(direct = 1))

        mps.canonicalize(direct = 1)
        # print(mps)
        self.assertTrue(mps.checkCanonical(direct = 1))
        self.assertFalse(mps.checkCanonical(direct = 0))
        