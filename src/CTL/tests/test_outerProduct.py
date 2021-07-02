from CTL.tests.packedTest import PackedTest

from CTL.tensor.tensor import Tensor 
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractTensorList
from CTL.tensor.contract.contract import contractTwoTensors
from CTL.tensor.diagonalTensor import DiagonalTensor
import CTL.funcs.funcs as funcs

import numpy as np 

class TestOuterProduct(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'OuterProduct')

    def test_outerProduct(self):
        a = Tensor(shape = (2, ), labels = ['a'])
        b = Tensor(shape = (2, ), labels = ['b'])

        op = contractTwoTensors(a, b, outProductWarning = False)
        self.assertTrue(funcs.compareLists(op.labels, ['a', 'b']))
        
        a = Tensor(shape = (2, 2, 2), labels = ['a', 'b', 'c'])
        b = Tensor(shape = (2, ), labels = ['x'])
        c = Tensor(shape = (2, ), labels = ['y'])

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)
        prod = contractTensorList([a, b, c], outProductWarning = False)
        self.assertTrue(funcs.compareLists(prod.labels, ['c']))

        dataA = np.random.random_sample((2, 2))
        dataB = np.random.random_sample((3, 3))
        a = Tensor(shape = (2, 2), labels = ['a1', 'a2'], data = dataA)
        b = Tensor(shape = (3, 3), labels = ['b1', 'b2'], data = dataB)

        prod = contractTensorList([a, b], outProductWarning = False)
        prod.reArrange(['a1', 'a2', 'b1', 'b2'])
        
        res = np.zeros((2, 2, 3, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    for l in range(3):
                        res[(i, j, k, l)] = dataA[(i, j)] * dataB[(k, l)]
        # print(res, prod.a)
        self.assertTrue(funcs.floatArrayEqual(res, prod.a))

        a = Tensor(shape = (2, 2), labels = ['a1', 'a2'], data = dataA)
        b = DiagonalTensor(shape = (3, 3), labels = ['b1', 'b2'], data = np.diag(dataB))
        prod = contractTensorList([a, b], outProductWarning = False)
        prodData = prod.toTensor(['a1', 'a2', 'b1', 'b2'])
        # prod.reArrange(['a1', 'a2', 'b1', 'b2'])
        
        res = np.zeros((2, 2, 3, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # for l in range(3):
                    res[(i, j, k, k)] = dataA[(i, j)] * dataB[(k, k)]
        # print(res, prod.a)
        self.assertTrue(funcs.floatArrayEqual(res, prodData))

        dataA = np.random.random_sample((2, 2))
        dataB = np.random.random_sample(3)

        a = Tensor(shape = (2, 2), labels = ['a1', 'a2'], data = dataA)
        b = DiagonalTensor(shape = (3, 3, 3), labels = ['b1', 'b2', 'b3'], data = dataB)
        prod = contractTensorList([a, b], outProductWarning = False)
        prodData = prod.toTensor(['a1', 'a2', 'b1', 'b2', 'b3'])
        # prod.reArrange(['a1', 'a2', 'b1', 'b2'])
        
        res = np.zeros((2, 2, 3, 3, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # for l in range(3):
                    res[(i, j, k, k, k)] = dataA[(i, j)] * dataB[k]
        # print(res, prod.a)
        self.assertTrue(funcs.floatArrayEqual(res, prodData))

        

        