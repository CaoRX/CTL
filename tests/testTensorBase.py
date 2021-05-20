import unittest 
from CTL.tensorbase.tensorbase import TensorBase
from CTL.tensor.tensor import Tensor
import numpy as np
from tests.packedTest import PackedTest

class TestTensorBase(PackedTest):

    def test_TensorBase(self):
        tensor = TensorBase(np.zeros((3, 4), dtype = np.float64))
        self.assertEqual(tensor.dim, 2)
        self.assertEqual(tensor.shape, (3, 4))
    
    def test_uninitializedTensorBase(self):
        tensor = TensorBase()
        self.assertTrue(tensor.dim is None)
        self.assertTrue(tensor.shape is None)
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'TensorBase')

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
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Tensor')

    # def test_uninitializedTensor(self):
    #     tensor = TensorBase()
    #     self.assertTrue(tensor.dim is None)
    #     self.assertTrue(tensor.shape is None)

if __name__ == '__main__':
    unittest.main()