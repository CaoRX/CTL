from CTL.tests.packedTest import PackedTest
from CTL.tensor.diagonalTensor import DiagonalTensor

from CTL.tensor.tensor import Tensor 
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractTensorList, generateOptimalSequence, contractCost
from CTL.tensor.contract.optimalContract import makeTensorGraph, contractAndCostWithSequence
import CTL.funcs.funcs as funcs
from CTL.tensor.leg import Leg

import numpy as np 

class TestDiagonalTensor(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'diagonalTensor')

    def test_diagonalTensor(self):
        self.showTestCaseBegin("diagonal tensor")
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
        self.showTestCaseEnd("diagonal tensor")

    def test_contraction(self):
        self.showTestCaseBegin("diagonal tensor contraction")
        # print('Begin test diagonalTensor contraction.')
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

        res1, _ = contractAndCostWithSequence([a, b, c])
        # print('seq = {}'.format(generateOptimalSequence([a, b, c])))

        a = Tensor(data = aData, labels = ['a', 'b', 'c'])
        b = Tensor(data = bData, labels = ['x'])
        c = Tensor(data = cData, labels = ['y'])

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)

        res2, _ = contractAndCostWithSequence([a, b, c])
        # self.assertListEqual(list(res1.a), list(res2.a))
        self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))

        # print(cost1, cost2)

        # print(res1.a, res2.a)

        # test diagonal tensor contraction
        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'])
        b = DiagonalTensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'])
        makeLink('a1', 'b1', a, b)
        res = a @ b
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
        # print(res)
        self.assertEqual(res.dim, 3)
        self.assertTrue(funcs.compareLists(list(res.shape), [2, 3, 3]))
        self.assertEqual(cost, 18.0)
        self.showTestCaseEnd("diagonal tensor contraction")

    def test_deduce(self):

        self.showTestCaseBegin("diagonal tensor shape deduction")

        legA = Leg(tensor = None, dim = 5, name = 'a')
        legB = Leg(tensor = None, dim = 5, name = 'b')
        legBError = Leg(tensor = None, dim = 6, name = 'b')

        a = DiagonalTensor(legs = [legA, legB])
        self.assertTupleEqual(a.shape, (5, 5))
        self.assertEqual(a.dim, 2)
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b']))
        self.assertTrue(funcs.floatArrayEqual(a.a, np.ones(5)))

        a = DiagonalTensor(legs = [legA, legB], labels = ['a', 'b'])
        self.assertTupleEqual(a.shape, (5, 5))
        self.assertEqual(a.dim, 2)
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b']))
        self.assertTrue(funcs.floatArrayEqual(a.a, np.ones(5)))

        a = DiagonalTensor(legs = [legA, legB], shape = 5)
        self.assertTupleEqual(a.shape, (5, 5))
        self.assertEqual(a.dim, 2)
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b']))
        self.assertTrue(funcs.floatArrayEqual(a.a, np.ones(5)))

        a = DiagonalTensor(legs = [legA, legB], shape = 5, data = np.zeros(5))
        self.assertTupleEqual(a.shape, (5, 5))
        self.assertEqual(a.dim, 2)
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b']))
        self.assertTrue(funcs.floatArrayEqual(a.a, np.zeros(5)))

        def legDimNotEqualFunc():
            _ = DiagonalTensor(legs = [legA, legBError])

        def labelsSizeNotEqualFunc():
            _ = DiagonalTensor(legs = [legA, legB], labels = ['a'])
        def labelsOrderNotEqualFunc():
            _ = DiagonalTensor(legs = [legA, legB], labels = ['b', 'a'])

        def shapeSizeNotEqualFunc():
            _ = DiagonalTensor(legs = [legA, legB], shape = (5, 6, 7))
        def shapeOrderNotEqualFunc():
            _ = DiagonalTensor(legs = [legA, legB], shape = (6, 5))
        
        def dataDimNotEqualFunc():
            _ = DiagonalTensor(legs = [legA, legB], data = np.zeros((5, 6, 7)))
        def dataShapeNotEqualFunc():
            _ = DiagonalTensor(legs = [legA, legB], data = np.zeros((5, 6)))
        def labelsShapeNotCompatibleFunc():
            _ = DiagonalTensor(legs = [legA, legB], labels = ['a'], data = np.zeros((5, 5)))
        def dimensionless1DDataErrorFunc():
            _ = DiagonalTensor(legs = [], labels = [], data = np.zeros(3))
        def dimensionless1DDataErrorFunc2():
            _ = DiagonalTensor(labels = [], data = np.zeros(3))

        self.assertRaises(ValueError, legDimNotEqualFunc)
        self.assertRaises(ValueError, labelsSizeNotEqualFunc)
        self.assertRaises(ValueError, labelsOrderNotEqualFunc)
        self.assertRaises(ValueError, shapeSizeNotEqualFunc)
        self.assertRaises(ValueError, shapeOrderNotEqualFunc)
        self.assertRaises(ValueError, dataDimNotEqualFunc)
        self.assertRaises(ValueError, dataShapeNotEqualFunc)
        self.assertRaises(ValueError, labelsShapeNotCompatibleFunc)
        self.assertRaises(ValueError, dimensionless1DDataErrorFunc)
        self.assertRaises(ValueError, dimensionless1DDataErrorFunc2)

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

        self.showTestCaseEnd("diagonal tensor shape deduction")

    def test_extremeContraction(self):

        self.showTestCaseBegin("diagonal tensor extreme contraction")

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 2))
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aData)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res1 = a @ b

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res2 = b @ a

        self.assertTrue(funcs.compareLists(list(res1.labels), ['a2']))
        self.assertTrue(funcs.compareLists(list(res2.labels), ['a2']))
        # self.assertListEqual(list(res1.a), list(res2.a))
        self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))

        aData = np.random.random_sample(2)
        aTensorData = np.array([[aData[0], 0], [0, aData[1]]])
        bData = np.random.random_sample((2, 2))
        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'], data = aData)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData)
        makeLink('a1', 'b1', a, b)
        makeLink('a2', 'b2', a, b)
        res1 = a @ b

        a = Tensor(shape = (2, 2), labels = ['a1', 'a2'], data = aTensorData)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData)
        makeLink('a1', 'b1', a, b)
        makeLink('a2', 'b2', a, b)
        res2 = b @ a

        self.assertTrue(funcs.compareLists(res1.labels, []))
        self.assertTrue(funcs.compareLists(res2.labels, []))
        # self.assertListEqual(list(res1.single(), list(res2.a))
        self.assertTrue(funcs.floatEqual(res1.single(), res2.single()))

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 2, 2))
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aData)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res1 = a @ b

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res2 = b @ a

        self.assertTrue(funcs.compareLists(list(res1.labels), ['a2', 'b3']))
        self.assertTrue(funcs.compareLists(list(res2.labels), ['a2', 'b3']))
        # print(res1.labels, res2.labels)
        # print(res1.a, res2.a)
        res2.reArrange(res1.labels)
        self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))
        # self.assertListEqual(list(np.ravel(res1.a)), list(np.ravel(res2.a)))

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 2, 2))
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aData)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res1 = a @ b

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res2 = b @ a

        # print('ndEye(2, 2) = {}'.format(funcs.ndEye(2, 2)))
        # print('ndEye(1, 2) = {}'.format(funcs.ndEye(1, 2)))
        # print('ndEye(3, 2) = {}'.format(funcs.ndEye(3, 2)))

        self.assertTrue(funcs.compareLists(list(res1.labels), ['a2', 'a3', 'b2', 'b3']))
        self.assertTrue(funcs.compareLists(list(res2.labels), ['a2', 'a3', 'b2', 'b3']))
        # print(res1.labels, res2.labels)
        res2.reArrange(res1.labels)
        # print('res1 = {}, res2 = {}'.format(res1.a, res2.a))
        self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))
        # self.assertListEqual(list(np.ravel(res1.a)), list(np.ravel(res2.a)))

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 4, 7))
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aData)
        b = Tensor(shape = (2, 4, 7), labels = ['b1', 'b2', 'b3'], data = bData)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res1 = a @ b

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData)
        b = Tensor(shape = (2, 4, 7), labels = ['b1', 'b2', 'b3'], data = bData)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res2 = b @ a

        # print('ndEye(2, 2) = {}'.format(funcs.ndEye(2, 2)))
        # print('ndEye(1, 2) = {}'.format(funcs.ndEye(1, 2)))
        # print('ndEye(3, 2) = {}'.format(funcs.ndEye(3, 2)))

        self.assertTrue(funcs.compareLists(list(res1.labels), ['a2', 'a3', 'b2', 'b3']))
        self.assertTrue(funcs.compareLists(list(res2.labels), ['a2', 'a3', 'b2', 'b3']))
        # print(res1.labels, res2.labels)
        res2.reArrange(res1.labels)
        # print('res1 = {}, res2 = {}'.format(res1.a, res2.a))
        self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))

        self.showTestCaseEnd("diagonal tensor extreme contraction")

    def test_toTensor(self):
        self.showTestCaseBegin("diagonal tensor toTensor")
        a = DiagonalTensor(shape = (3, 3, 3), labels = ['a', 'b', 'c'], data = np.array([1, 2, 3]))
        aTensor = a.toTensor()
        aRealTensor = np.zeros((3, 3, 3))
        for i in range(3):
            aRealTensor[(i, i, i)] = i + 1
        self.assertTrue(funcs.floatArrayEqual(aTensor, aRealTensor))
        self.showTestCaseEnd("diagonal tensor toTensor")

    def test_DiagonalTensorCopy(self):
        self.showTestCaseBegin("diagonal tensor copy")
        aData = np.ones((3, 3))
        a = DiagonalTensor(shape = (3, 3), labels = ['a', 'b'], data = aData)
        aData[(0, 0)] = 2.0
        self.assertEqual(a.a[0], 1.0)

        a = DiagonalTensor(shape = (3, 3), labels = ['a', 'b'])
        b = a.copy()
        b.a[0] = 2.0
        self.assertEqual(a.a[0], 1.0)
        self.assertEqual(b.a[0], 2.0)
        self.assertEqual(a.diagonalFlag, b.diagonalFlag)
        self.assertEqual(a.tensorLikeFlag, b.tensorLikeFlag)

        a.renameLabel('a', 'c')
        self.assertListEqual(b.labels, ['a', 'b'])
        self.showTestCaseEnd('diagonal tensor copy')

class TestDiagonalTensorLike(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Diagonal TensorLike')
    
    def test_diagonalTensorLike(self):
        diagonalTensor = DiagonalTensor(shape = (3, 3), tensorLikeFlag = True)
        self.assertEqual(diagonalTensor.dim, 2)
        self.assertTupleEqual(diagonalTensor.shape, (3, 3))
        self.assertListEqual(diagonalTensor.labels, ['a', 'b'])
        self.assertIsNone(diagonalTensor.a)

        diagonalTensor = DiagonalTensor(data = np.zeros((3, 3)), tensorLikeFlag = True) # only data is given
        self.assertTrue(diagonalTensor.diagonalFlag)
        self.assertEqual(diagonalTensor.dim, 2)
        self.assertTupleEqual(diagonalTensor.shape, (3, 3))
        self.assertListEqual(diagonalTensor.labels, ['a', 'b'])
        self.assertIsNone(diagonalTensor.a)

        diagonalTensor = DiagonalTensor(data = np.ones((3, 3)), labels = ['up', 'down'], tensorLikeFlag = True) # only data is given
        self.assertTrue(diagonalTensor.diagonalFlag)
        self.assertEqual(diagonalTensor.dim, 2)
        self.assertTupleEqual(diagonalTensor.shape, (3, 3))
        self.assertListEqual(diagonalTensor.labels, ['up', 'down'])
        self.assertIsNone(diagonalTensor.a)

        self.assertRaises(AssertionError, diagonalTensor.norm)
        self.assertRaises(AssertionError, diagonalTensor.trace)
        self.assertRaises(AssertionError, diagonalTensor.toTensor)
        self.assertRaises(AssertionError, lambda: diagonalTensor.toMatrix(rows = None, cols = None))
        self.assertRaises(AssertionError, diagonalTensor.toVector)

    def test_toTensorLike(self):
        a = DiagonalTensor(shape = (3, 3), tensorLikeFlag = False)
        self.assertIsNotNone(a.a)

        aLike = a.toTensorLike()
        self.assertIsNone(aLike.a)
        self.assertTrue(aLike.tensorLikeFlag)
        self.assertTupleEqual(aLike.shape, (3, 3))
        self.assertListEqual(aLike.labels, ['a', 'b'])

    def test_diagonalRename(self):

        a = DiagonalTensor(shape = (3, 3, 3), labels = ['abc', 'def', 'abc'])
        a.renameLabels(['abc', 'abc'], ['abc1', 'abc2'])
        self.assertTrue(funcs.compareLists(a.labels, ['abc1', 'abc2', 'def']))

    def test_diagonalSumOut(self):
        a = DiagonalTensor(shape = (3, 3, 3), labels = ['abc', 'def', 'abc'])
        a.sumOutLegByLabel(['abc', 'abc'])
        self.assertTrue(funcs.compareLists(a.labels, ['def']))






        


