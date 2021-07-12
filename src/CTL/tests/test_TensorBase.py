import unittest 
from CTL.tensorbase.tensorbase import TensorBase
import numpy as np
from CTL.tests.packedTest import PackedTest

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