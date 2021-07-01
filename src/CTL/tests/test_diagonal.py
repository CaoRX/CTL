from CTL.tests.packedTest import PackedTest
from CTL.tensor.diagonalTensor import DiagonalTensor

from CTL.tensor.tensor import Tensor 
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractTensorList, generateOptimalSequence, contractCost
from CTL.tensor.contract.contract import contractTwoTensors
from CTL.tensor.contract.optimalContract import makeTensorGraph, contractAndCostWithSequence
import CTL.funcs.funcs as funcs

import numpy as np 

class TestDiagonalTensor(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'diagonalTensor')

    def test_diagonalTensor(self):
        diagonalTensor = DiagonalTensor(shape = (3, 3))
        self.assertEqual(diagonalTensor.dim, 2)
        self.assertTupleEqual(diagonalTensor.shape, (3, 3))
        self.assertListEqual(diagonalTensor.labels, ['a', 'b'])

        diagonalTensor = DiagonalTensor(data = np.zeros((3, 3))) # only data is given
        self.assertTrue(diagonalTensor.diagonalFlag)
        self.assertEqual(diagonalTensor.dim, 2)
        self.assertTupleEqual(diagonalTensor.shape, (3, 3))
        self.assertListEqual(diagonalTensor.labels, ['a', 'b'])

        diagonalTensor = DiagonalTensor(data = np.ones((3, 3)), labels = ['up', 'down']) # only data is given
        self.assertTrue(diagonalTensor.diagonalFlag)
        self.assertEqual(diagonalTensor.dim, 2)
        self.assertTupleEqual(diagonalTensor.shape, (3, 3))
        self.assertListEqual(diagonalTensor.labels, ['up', 'down'])

        self.assertEqual(diagonalTensor.norm(), np.sqrt(3.0))
        self.assertEqual(diagonalTensor.trace(), 3.0)

    def test_contraction(self):
        print('Begin test diagonalTensor contraction.')
        a = DiagonalTensor(shape = (2, 2), labels = ['a', 'b'])
        b = Tensor(shape = (2, ), labels = ['x'])
        c = Tensor(shape = (2, ), labels = ['y'])

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)
        seq = generateOptimalSequence([a, b, c], typicalDim = 10)
        # print('optimal sequence = {}'.format(seq))
        prod, cost = contractAndCostWithSequence([a, b, c], seq = seq)
        # print('cost = {}'.format(cost))
        # prod = contractTensorList([a, b, c], outProductWarning = False)
        self.assertTrue(funcs.compareLists(prod.labels, []))
        self.assertListEqual(seq, [(0, 2), (1, 0)])
        self.assertEqual(cost, 4.0)

        # if we use Tensor instead of DiagonalTensor for a
        # then the cost should be 12.0, and the order should be (1, 2), (0, 1)
        # the optimal cost of diagonal tensors can be achieved if we use diagonal nature for contraction

        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a', 'b', 'c'])
        b = DiagonalTensor(shape = (2, 2), labels = ['x', 'y'])
        makeLink('a', 'x', a, b)
        prod, cost = contractAndCostWithSequence([a, b])
        self.assertEqual(cost, 2)
        self.assertTrue(funcs.compareLists(prod.labels, ['b', 'c', 'y']))

        aData = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 3]]])
        bData = np.random.random_sample(2)
        cData = np.random.random_sample(2)

        a = DiagonalTensor(data = aData, labels = ['a', 'b', 'c'])
        b = Tensor(data = bData, labels = ['x'])
        c = Tensor(data = cData, labels = ['y'])

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)

        res1, cost1 = contractAndCostWithSequence([a, b, c])
        print('seq = {}'.format(generateOptimalSequence([a, b, c])))

        a = Tensor(data = aData, labels = ['a', 'b', 'c'])
        b = Tensor(data = bData, labels = ['x'])
        c = Tensor(data = cData, labels = ['y'])

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)

        res2, cost2 = contractAndCostWithSequence([a, b, c])
        self.assertListEqual(list(res1.a), list(res2.a))

        # print(cost1, cost2)

        # print(res1.a, res2.a)

        # test diagonal tensor contraction
        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'])
        b = DiagonalTensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'])
        makeLink('a1', 'b1', a, b)
        res = contractTwoTensors(a, b)
        self.assertTupleEqual(res.shape, (2, 2, 2))
        self.assertEqual(res.dim, 3)
        self.assertTrue(res.diagonalFlag)
        self.assertTrue((res.a == np.ones(2)).all())

        # test for diagonal * diagonal contraction cost(just O(length))
        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'])
        b = DiagonalTensor(shape = 2, labels = ['b1', 'b2']) # deduce dim
        makeLink('a1', 'b2', a, b)

        cost, _ = contractCost(a, b)
        self.assertEqual(cost, 2.0)

        res, cost = contractAndCostWithSequence([a, b])
        self.assertEqual(res.dim, 2)
        self.assertEqual(res._length, 2)
        self.assertTupleEqual(res.shape, (2, 2))
        self.assertEqual(cost, 2.0)
        self.assertTrue(res.diagonalFlag)

        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'])
        b = Tensor(shape = (2, 3, 3), labels = ['b1', 'b2', 'b3']) # deduce dim
        makeLink('a1', 'b1', a, b)

        cost, _ = contractCost(a, b)
        self.assertEqual(cost, 18.0)

        res, cost = contractAndCostWithSequence([a, b])
        print(res)
        self.assertEqual(res.dim, 3)
        self.assertTrue(funcs.compareLists(list(res.shape), [2, 3, 3]))
        self.assertEqual(cost, 18.0)



    def test_deduce(self):

        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'])
        self.assertTupleEqual(a.shape, (2, 2))
        self.assertEqual(a.dim, 2)

        a = DiagonalTensor(shape = (2, 2))
        self.assertTupleEqual(a.shape, (2, 2))
        self.assertEqual(a.dim, 2)
        self.assertTrue((a.a == np.ones(2)).all()) # default as identity tensor

        def shapeNotEqualFunc():
            _ = DiagonalTensor(shape = (2, 3))
        
        self.assertRaises(ValueError, shapeNotEqualFunc)

        def labelsShortFunc():
            _ = DiagonalTensor(shape = (2, 2), labels = ['a1'])
        def labelsLongFunc():
            _ = DiagonalTensor(shape = (2, 2), labels = ['a', 'b', 'c'])

        self.assertRaises(ValueError, labelsShortFunc)
        self.assertRaises(ValueError, labelsLongFunc)

        a = DiagonalTensor(shape = (2, 2), data = np.zeros((2, 2)))
        self.assertTupleEqual(a.shape, (2, 2))
        self.assertEqual(a.dim, 2)

        a = DiagonalTensor(shape = (2, 2, 2), data = np.zeros(2))
        self.assertTupleEqual(a.shape, (2, 2, 2))
        self.assertEqual(a.dim, 3)

        def dataDimErrorFunc():
            _ = DiagonalTensor(shape = (2, 2), data = np.zeros((2, 2, 2)))
        def dataShapeErrorFunc():
            _ = DiagonalTensor(shape = (2, 2), data = np.zeros((2, 3))) # no error in 9be9325, newly added

        self.assertRaises(ValueError, dataDimErrorFunc)
        self.assertRaises(ValueError, dataShapeErrorFunc)

        # now start (shape = None) tests

        a = DiagonalTensor(labels = ['a', 'b'], data = np.zeros(3))
        self.assertEqual(a._length, 3)
        self.assertEqual(a.shape, (3, 3))
        self.assertEqual(a.dim, 2)

        a = DiagonalTensor(labels = ['a', 'b'], data = np.zeros((4, 4)))
        self.assertEqual(a._length, 4)
        self.assertEqual(a.shape, (4, 4))
        self.assertEqual(a.dim, 2)

        a = DiagonalTensor(labels = ['a', 'b'], data = np.array([[1, 2], [3, 4]]))
        self.assertEqual(a._length, 2)
        self.assertEqual(a.shape, (2, 2))
        self.assertTrue((a.a == np.array([1, 4])).all())

        def dataDimErrorFunc2():
            _ = DiagonalTensor(labels = ['a', 'b', 'c'], data = np.zeros((2, 2)))
        
        def dataShapeErrorFunc2():
            _ = DiagonalTensor(labels = ['a', 'b', 'c'], data = np.zeros((2, 2, 3)))
        
        def dataNoneErrorFunc():
            _ = DiagonalTensor(labels = ['a', 'b', 'c'], data = None)
        
        self.assertRaises(ValueError, dataDimErrorFunc2)
        self.assertRaises(ValueError, dataShapeErrorFunc2)
        self.assertRaises(ValueError, dataNoneErrorFunc)

        # now start(shape = None, labels = None)
        
        a = DiagonalTensor(data = np.zeros(2)) # 1D diagonal tensor as a simple vector
        self.assertEqual(a.dim, 1)
        self.assertEqual(a._length, 2)
        self.assertListEqual(a.labels, ['a'])

        a = DiagonalTensor(data = np.array([[1, 2], [4, 3]]))
        self.assertEqual(a.dim, 2)
        self.assertEqual(a._length, 2)
        self.assertTrue((a.a == np.array([1, 3])).all())

        def dataShapeErrorFunc3():
            _ = DiagonalTensor(data = np.zeros((2, 2, 3)))
        
        def nothingErrorFunc():
            _ = DiagonalTensor()

        self.assertRaises(ValueError, dataShapeErrorFunc3)
        self.assertRaises(ValueError, nothingErrorFunc)

        


