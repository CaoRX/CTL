import unittest 
from CTL.tensor.tensor import Tensor
import numpy as np
from CTL.tests.packedTest import PackedTest
from CTL.tensor.leg import Leg
import CTL.funcs.funcs as funcs

class TestTensor(PackedTest):
    def test_Tensor(self):
        data = np.zeros((3, 4), dtype = np.float64)
        tensor = Tensor(data = data)
        self.assertEqual(tensor.dim, 2)
        self.assertEqual(tensor.shape, (3, 4))
        self.assertListEqual(tensor.labels, ['a', 'b'])

        data[0][0] = 1.0
        self.assertEqual(tensor.a[0][0], 0)

    def test_TensorLabels(self):
        tensor = Tensor(data = np.zeros((3, 4, 5), dtype = np.float64), labels = ['abc', 'def', 'abc'])
        self.assertEqual(tensor.indexOfLabel('abc'), 0)
        self.assertEqual(tensor.indexOfLabel('abd'), -1)
        self.assertEqual(tensor.indexOfLabel('abc', backward = True), 2)

        self.assertEqual(tensor.shapeOfLabels(['abc', 'def']), (3, 4))

    def test_TensorDeduction(self):
        # test the deduction of Tensor

        # deduce strategy:
        # we want shape, and labels
        # we have legs, shape, labels, data
        # priority for shape: legs > shape > data
        # priority for labels: legs > labels

        # 1. legs exist: 
        # if labels exist: check the length and content of labels with legs
        # if shape exist: check whether shape == tuple([leg.dim for leg in legs])
        # if data exist: check whehter data.shape == tuple([leg.dim for leg in legs]) (if not, but the total size equal, we transfer data to the given shape)
        
        # 3. legs not exist, shape not exist
        # if data exist: generate shape according to data, and auto-generate legs

        legA = Leg(tensor = None, dim = 5, name = 'a')
        legB = Leg(tensor = None, dim = 6, name = 'b')

        a = Tensor(legs = [legA, legB])
        self.assertTupleEqual(a.shape, (5, 6))
        self.assertEqual(a.dim, 2)
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b']))

        a = Tensor(legs = [legA, legB], labels = ['a', 'b'])
        self.assertTupleEqual(a.shape, (5, 6))
        self.assertEqual(a.dim, 2)
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b']))

        def labelsSizeNotEqualFunc():
            _ = Tensor(legs = [legA, legB], labels = ['a'])
        def labelsOrderNotEqualFunc():
            _ = Tensor(legs = [legA, legB], labels = ['b', 'a'])

        def shapeSizeNotEqualFunc():
            _ = Tensor(legs = [legA, legB], shape = (5, 6, 7))
        def shapeOrderNotEqualFunc():
            _ = Tensor(legs = [legA, legB], shape = (6, 5))
        
        def dataDimNotEqualFunc():
            _ = Tensor(legs = [legA, legB], data = np.zeros((5, 6, 7)))
        def dataShapeNotEqualFunc():
            _ = Tensor(legs = [legA, legB], data = np.zeros((5, 5)))

        self.assertRaises(ValueError, labelsSizeNotEqualFunc)
        self.assertRaises(ValueError, labelsOrderNotEqualFunc)
        self.assertRaises(ValueError, shapeSizeNotEqualFunc)
        self.assertRaises(ValueError, shapeOrderNotEqualFunc)
        self.assertRaises(ValueError, dataDimNotEqualFunc)
        self.assertRaises(ValueError, dataShapeNotEqualFunc)

        # for data, works if (data size) = (dim product), no matter what the shape of data

        a = Tensor(legs = [legA, legB], data = np.zeros((6, 5)))
        self.assertTupleEqual(a.shape, (5, 6))
        self.assertEqual(a.dim, 2)
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b']))

        a = Tensor(legs = [legA, legB], data = np.zeros((3, 2, 5)))
        self.assertTupleEqual(a.shape, (5, 6))
        self.assertEqual(a.dim, 2)
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b']))

        a = Tensor(legs = [legA, legB], data = np.zeros(30))
        self.assertTupleEqual(a.shape, (5, 6))
        self.assertEqual(a.dim, 2)
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b']))

        # 2. legs not exist, shape exist
        # if data exist, check the total number of components of data equal to shape, otherwise random
        # if labels exist: check the number of labels equal to dimension, otherwise auto-generate

        a = Tensor(shape = (5, 3, 4))
        self.assertTupleEqual(a.shape, (5, 3, 4))
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b', 'c']))
        self.assertEqual(a.dim, 3)

        a = Tensor(shape = (5, 3, 4), data = np.zeros(60))
        self.assertTupleEqual(a.shape, (5, 3, 4))
        self.assertTrue(funcs.compareLists(a.labels, ['a', 'b', 'c']))
        self.assertEqual(a.dim, 3)
        self.assertEqual(a.a[(0, 0, 0)], 0)

        a = Tensor(shape = (5, 3, 4), labels = ['c', 'b', 'a'])
        self.assertTupleEqual(a.shape, (5, 3, 4))
        self.assertTrue(funcs.compareLists(a.labels, ['c', 'b', 'a']))
        self.assertEqual(a.dim, 3)

        a = Tensor(shape = (5, 3, 4), labels = ('c', 'b', 'a')) # tuple also is acceptable
        self.assertTupleEqual(a.shape, (5, 3, 4))
        self.assertTrue(funcs.compareLists(a.labels, ['c', 'b', 'a']))
        self.assertEqual(a.dim, 3)

        a = Tensor(shape = (5, 3, 4), labels = ['c', 'b', 'a'], data = np.zeros(60))
        self.assertTupleEqual(a.shape, (5, 3, 4))
        self.assertTrue(funcs.compareLists(a.labels, ['c', 'b', 'a']))
        self.assertEqual(a.dim, 3)

        def dataTotalSizeNotEqualErrorFunc():
            _ = Tensor(shape = (5, 3, 4), data = np.zeros((5, 3, 3)))
        def labelsSizeNotEqualFunc2():
            _ = Tensor(shape = (5, 3, 4), labels = ['a', 'b'])

        self.assertRaises(ValueError, dataTotalSizeNotEqualErrorFunc)
        self.assertRaises(ValueError, labelsSizeNotEqualFunc2)

        # 3. legs not exist, shape not exist
        # if data exist: generate shape according to data, and auto-generate legs

        a = Tensor(data = np.zeros((3, 4, 5)))
        self.assertTupleEqual(a.shape, (3, 4, 5))
        self.assertEqual(a.dim, 3)
        self.assertEqual(a.legs[0].dim, 3)
        self.assertEqual(a.legs[0].name, 'a')

        def nothingErrorFunc():
            _ = Tensor()

        self.assertRaises(ValueError, nothingErrorFunc)

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Tensor')