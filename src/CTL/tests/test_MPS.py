import unittest 
from CTL.tensor.tensor import Tensor
import numpy as np
from CTL.tests.packedTest import PackedTest
import CTL.funcs.funcs as funcs
from CTL.examples.MPS import FreeBoundaryMPS
from CTL.tensor.contract.link import makeLink
from CTL.tensor.tensorFunc import isIsometry

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

        self.assertTrue(mps.checkMPSProperty(mps._tensors))

        mps.canonicalize(idx = 0)
        # print(mps)
        mps.canonicalize(idx = 0)
        # print(mps)

        self.assertTrue(mps.checkCanonical(excepIdx = 0))
        self.assertFalse(mps.checkCanonical(excepIdx = 2))

        mps.canonicalize(idx = 2)
        # print(mps)
        self.assertTrue(mps.checkCanonical(excepIdx = 2))
        self.assertFalse(mps.checkCanonical(excepIdx = 0))
        self.assertTrue(mps.checkCanonical())
        self.assertEqual(mps.activeIdx, 2)

        mps.moveTensor(2, 4)
        self.assertTrue(isIsometry(mps.getTensor(0), labels = ['o']))
        self.assertTrue(isIsometry(mps.getTensor(1), labels = ['o', 'l']))

    def test_singleTensorMPS(self):
        tensor = Tensor(data = np.random.random_sample(3), labels = ['oo'])
        mps = FreeBoundaryMPS(tensorList = [tensor], chi = 16)
        
        self.assertEqual(mps.n, 1)
        mps.canonicalize(0)
        self.assertTrue(mps.checkCanonical(excepIdx = 0))
        self.assertEqual(mps.getTensor(0).legs[0].name, 'o')