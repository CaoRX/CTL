import unittest 
from CTL.tensor.tensor import Tensor
import numpy as np
from CTL.tests.packedTest import PackedTest
import CTL.funcs.funcs as funcs
from CTL.examples.MPS import FreeBoundaryMPS, mergeMPS, contractMPS
from CTL.tensor.contract.link import makeLink
from CTL.tensor.tensorFunc import isIsometry

class TestMPS(PackedTest):
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'MPS')
    
    def test_MPS(self):
        
        # tensorL = Tensor(data = np.random.random_sample((3, 3)), labels = ['o', 'internal'])
        # tensor1 = Tensor(data = np.random.random_sample((3, 5, 4)), labels = ['itl', 'oo', 'itr'])
        # tensor2 = Tensor(data = np.random.random_sample((4, 2, 4)), labels = ['itl', 'oo', 'itr'])
        # tensor3 = Tensor(data = np.random.random_sample((4, 3, 2)), labels = ['itl', 'oo', 'itr'])
        # tensorR = Tensor(data = np.random.random_sample((2, 5)), labels = ['internal', 'o'])

        # makeLink('internal', 'itl', tensorL, tensor1)
        # makeLink('itr', 'itl', tensor1, tensor2)
        # makeLink('itr', 'itl', tensor2, tensor3)
        # makeLink('itr', 'internal', tensor3, tensorR)

        # tensors = [tensorL, tensor1, tensor2, tensor3, tensorR]

        # mps = FreeBoundaryMPS(tensorList = tensors, chi = 16)
        mps = self.createMPSA()
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

    def createMPSA(self, tensorLikeFlag = False):
        tensor1L = Tensor(shape = (3, 3), labels = ['o', 'internal'], tensorLikeFlag = tensorLikeFlag)
        tensor11 = Tensor(shape = (3, 5, 4), labels = ['itl', 'oo', 'itr'], tensorLikeFlag = tensorLikeFlag)
        tensor12 = Tensor(shape = (4, 2, 4), labels = ['itl', 'oo', 'itr'], tensorLikeFlag = tensorLikeFlag)
        tensor13 = Tensor(shape = (4, 3, 2), labels = ['itl', 'oo', 'itr'], tensorLikeFlag = tensorLikeFlag)
        tensor1R = Tensor(shape = (2, 5), labels = ['internal', 'o'], tensorLikeFlag = tensorLikeFlag)

        makeLink('internal', 'itl', tensor1L, tensor11)
        makeLink('itr', 'itl', tensor11, tensor12)
        makeLink('itr', 'itl', tensor12, tensor13)
        makeLink('itr', 'internal', tensor13, tensor1R)

        tensorsA = [tensor1L, tensor11, tensor12, tensor13, tensor1R]

        mpsA = FreeBoundaryMPS(tensorList = tensorsA, chi = 16)
        return mpsA

    def createMPSB(self, tensorLikeFlag = False):
        tensor2L = Tensor(shape = (3, 3), labels = ['o', 'internal'], tensorLikeFlag = tensorLikeFlag)
        tensor21 = Tensor(shape = (3, 5, 4), labels = ['itl', 'oo', 'itr'], tensorLikeFlag = tensorLikeFlag)
        tensor22 = Tensor(shape = (4, 2, 4), labels = ['itl', 'oo', 'itr'], tensorLikeFlag = tensorLikeFlag)
        tensor2R = Tensor(shape = (4, 5), labels = ['internal', 'o'], tensorLikeFlag = tensorLikeFlag)

        makeLink('internal', 'itl', tensor2L, tensor21)
        makeLink('itr', 'itl', tensor21, tensor22)
        makeLink('itr', 'internal', tensor22, tensor2R)

        tensorsB = [tensor2L, tensor21, tensor22, tensor2R]
        mpsB = FreeBoundaryMPS(tensorList = tensorsB, chi = 12)
        return mpsB

    def createMPSFromDim(self, dims, itbRange = (3, 10), tensorLikeFlag = False, chi = 16):
        # internal bonds will be automaticall
        lastDim = -1
        tensors = []
        n = len(dims)
        
        if (n == 1):
            tensors.append(Tensor(shape = (dims[0], ), labels = ['o'], tensorLikeFlag = tensorLikeFlag))
            return FreeBoundaryMPS(tensorList = tensors, chi = chi)

        itbLow, itbHigh = itbRange
        
        bondDim = np.random.randint(low = itbLow, high = itbHigh)
        tensor = Tensor(shape = (dims[0], bondDim), labels = ['o', 'r'], tensorLikeFlag = tensorLikeFlag)
        tensors.append(tensor)
        lastDim = bondDim 
        for i in range(1, n - 1):
            bondDim = np.random.randint(low = itbLow, high = itbHigh)
            newTensor = Tensor(shape = (lastDim, dims[i], bondDim), labels = ['l', 'o', 'r'], tensorLikeFlag = tensorLikeFlag)
            tensors.append(newTensor)
            makeLink('r', 'l', tensor, newTensor)
            lastDim = bondDim 
            tensor = newTensor
        
        newTensor = Tensor(shape = (lastDim, dims[-1]), labels = ['l', 'o'], tensorLikeFlag = tensorLikeFlag)
        tensors.append(newTensor)
        makeLink('r', 'l', tensor, newTensor) 

        return FreeBoundaryMPS(tensorList = tensors, chi = chi)
        
    def test_MPSContraction(self):
        mpsA = self.createMPSA(tensorLikeFlag = False)
        mpsB = self.createMPSB(tensorLikeFlag = False)

        tensorA2 = mpsA.getTensor(2)
        tensorB2 = mpsB.getTensor(2)

        makeLink('o', 'o', tensorA2, tensorB2)
        # print(mpsA, mpsB)
        mps = contractMPS(mpsA, mpsB)
        # print(mps)

        mps.canonicalize(idx = 2)
        self.assertTrue(mps.checkCanonical())
        self.assertTrue(mps.n, 7) # 4 + 5 - 2
        self.assertTrue(mps.activeIdx, 2)

        mpsA = self.createMPSA(tensorLikeFlag = True)
        mpsB = self.createMPSB(tensorLikeFlag = True)

        tensorA2 = mpsA.getTensor(2)
        tensorB2 = mpsB.getTensor(2)

        makeLink('o', 'o', tensorA2, tensorB2)
        # print(mpsA, mpsB)
        mps = contractMPS(mpsA, mpsB)
        # print(mps)

        mps.canonicalize(idx = 2)
        self.assertTrue(mps.checkCanonical())
        self.assertTrue(mps.n, 7) # 4 + 5 - 2
        self.assertTrue(mps.activeIdx, 2)


    def test_MPSMerge(self):
        mpsA = self.createMPSA()
        mpsB = self.createMPSB()

        makeLink('o', 'o', mpsA.getTensor(1), mpsB.getTensor(1))
        makeLink('o', 'o', mpsA.getTensor(4), mpsB.getTensor(3))

        print(mpsA, mpsB)

        mergeMPS(mpsA, mpsB)
        print(mpsA, mpsB)

        self.assertTrue(mpsA.checkCanonical(excepIdx = mpsA.n - 1))
        self.assertTrue(mpsB.checkCanonical(excepIdx = mpsB.n - 1))
        self.assertEqual(mpsA.n, 4)
        self.assertEqual(mpsB.n, 3)

        mpsA.moveTensor(mpsA.n - 1, 1)
        mpsB.moveTensor(mpsB.n - 1, 0)
        mps = contractMPS(mpsB, mpsA)
        mps.canonicalize(idx = 2)

        self.assertTrue(mps.checkCanonical())
        self.assertEqual(mps.n, 5)

        mpsA = self.createMPSFromDim(dims = [3, 4, 5, 5, 2])
        mpsB = self.createMPSFromDim(dims = [2, 5, 3, 3, 4])
        mpsA.canonicalize(idx = 1)
        mpsB.canonicalize(idx = 2)
        print(mpsA, mpsB)

        makeLink('o', 'o', mpsA.getTensor(1), mpsB.getTensor(4))
        makeLink('o', 'o', mpsA.getTensor(0), mpsB.getTensor(2))
        makeLink('o', 'o', mpsB.getTensor(0), mpsA.getTensor(4))

        mergeMPS(mpsA, mpsB, beginFlag = True)
        print(mpsA, mpsB)

        self.assertEqual(mpsA.n, 3)
        self.assertEqual(mpsB.n, 3) 
        mps = contractMPS(mpsA, mpsB)
        self.assertEqual(mps.n, 4)
        print(mps)