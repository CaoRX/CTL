import unittest 
from CTL.tensorbase.tensorbase import TensorBase
import numpy as np
from CTL.tests.packedTest import PackedTest
from CTL.tensor.tensor import Tensor
from CTL.tensor.tensorFunc import tensorSVDDecomposition
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractWithSequence

class TestTensorFuncs(PackedTest):
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'TensorFuncs')

    def test_tensorSVDDecomposition(self):
        data = np.random.randn(3, 4)

        a = Tensor(data = data, labels = ['A', 'B'])

        decomp = tensorSVDDecomposition(a, rows = ['A'], cols = ['B'])
        u, s, v = decomp['u'], decomp['s'], decomp['v']
        # print(u, s, v)

        # print(u.legs, v.legs)

        for leg in u.legs:
            self.assertIsNone(leg.bond)
        for leg in v.legs:
            self.assertIsNone(leg.bond)
        for leg in s.legs:
            self.assertIsNone(leg.bond)

        uTensor = u.toTensor(['A', 'inner:A'])
        vTensor = v.toTensor(['inner:B', 'B'])
        sTensor = s.toTensor(['A', 'B'])

        self.assertEqual(uTensor.shape, (3, 3))
        self.assertEqual(vTensor.shape, (3, 4))
        self.assertEqual(sTensor.shape, (3, 3))

        # Test SVD truncation with finite chi
        decomp = tensorSVDDecomposition(a, rows = ['A'], cols = ['B'], chi = 2)
        u, s, v = decomp['u'], decomp['s'], decomp['v']

        uTensor = u.toTensor(['A', 'inner:A'])
        vTensor = v.toTensor(['inner:B', 'B'])
        sTensor = s.toTensor(['A', 'B'])

        self.assertEqual(uTensor.shape, (3, 2))
        self.assertEqual(vTensor.shape, (2, 4))
        self.assertEqual(sTensor.shape, (2, 2))

        # After tensorSVDDecomposition(keepdim = False, preserveLegs = False), legs are not changed
        for leg in a.legs:
            self.assertEqual(leg.tensor, a)

        data = np.random.randn(3, 4, 5, 6)
        a = Tensor(data = data, labels = ['A', 'B', 'C', 'D'])

        chi = 10
        decomp = tensorSVDDecomposition(a, rows = ['A', 'C'], cols = ['B', 'D'], chi = chi)
        u, s, v = decomp['u'], decomp['s'], decomp['v']

        uTensor = u.toTensor(['A|C', 'inner:A|C'])
        vTensor = v.toTensor(['inner:B|D', 'B|D'])
        sTensor = s.toTensor(['A|C', 'B|D'])

        self.assertEqual(uTensor.shape, (15, chi))
        self.assertEqual(vTensor.shape, (chi, 24))
        self.assertEqual(sTensor.shape, (chi, chi))

        decomp = tensorSVDDecomposition(a, rows = ['A', 'C'], cols = ['B', 'D'], chi = 16)
        # if chi > max(row_shape, col_shape), then it will be automatically truncated
        u, s, v = decomp['u'], decomp['s'], decomp['v']

        uTensor = u.toTensor(['A|C', 'inner:A|C'])
        vTensor = v.toTensor(['inner:B|D', 'B|D'])
        sTensor = s.toTensor(['A|C', 'B|D'])

        self.assertEqual(uTensor.shape, (15, 15))
        self.assertEqual(vTensor.shape, (15, 24))
        self.assertEqual(sTensor.shape, (15, 15))

        makeLink('inner:A|C', 'A|C', u, s)
        makeLink('inner:B|D', 'B|D', v, s)

        aTruncated = contractWithSequence([u, v, s])

        aMat = a.toMatrix(rows = ['A', 'C'], cols = ['B', 'D'])
        aTruncatedMat = aTruncated.toMatrix(rows = ['A|C'], cols = ['B|D'])
        diff = np.linalg.norm(aMat - aTruncatedMat)
        self.assertTrue(np.abs(diff) < 1e-10)

        a1 = a.copy()
        decomp = tensorSVDDecomposition(a1, rows = ['A', 'C'], cols = ['B', 'D'], chi = 16, keepdim = True)
        u, s, v = decomp['u'], decomp['s'], decomp['v']

        for leg in u.legs:
            self.assertIsNone(leg.bond)
        for leg in v.legs:
            self.assertIsNone(leg.bond)
        for leg in s.legs:
            self.assertIsNone(leg.bond)
        # print(u, s, v)

        # even after keepdim = True SVD, legs are still from a1
        for leg in a1.legs:
            self.assertEqual(leg.tensor, a1)
        
        self.assertEqual(u.shapeOfLabels(['A', 'C', 'inner:A|C']), (3, 5, 15))
        self.assertEqual(v.shapeOfLabels(['D', 'B', 'inner:B|D']), (6, 4, 15))

        makeLink('inner:A|C', 'A|C', u, s)
        makeLink('inner:B|D', 'B|D', v, s)

        a1Truncated = contractWithSequence([u, s, v])

        a1Mat = a1.toTensor(['A', 'B', 'C', 'D'])
        a1TruncatedMat = a1Truncated.toTensor(['A', 'B', 'C', 'D'])
        diff = np.linalg.norm(a1Mat - a1TruncatedMat)
        self.assertTrue(np.abs(diff) < 1e-10)

        a1 = a.copy()
        decomp = tensorSVDDecomposition(a1, rows = ['A', 'C'], cols = ['B', 'D'], chi = 16, keepdim = True, preserveLegs = True)
        u, s, v = decomp['u'], decomp['s'], decomp['v']
        # print(u, s, v)

        # even after keepdim = True SVD, legs are still from a1
        for leg in a1.legs:
            self.assertNotEqual(leg.tensor, a1)
        
        for label in ['A', 'C']:
            self.assertEqual(a1.getLeg(label).tensor, u)
        for label in ['B', 'D']:
            self.assertEqual(a1.getLeg(label).tensor, v)

        for leg in u.legs:
            self.assertIsNone(leg.bond)
        for leg in v.legs:
            self.assertIsNone(leg.bond)
        for leg in s.legs:
            self.assertIsNone(leg.bond)

        makeLink('inner:A|C', 'A|C', u, s)
        makeLink('inner:B|D', 'B|D', v, s)
        a1Truncated = contractWithSequence([u, s, v])

        aMat = a.toTensor(['A', 'B', 'C', 'D']) # do not use a1, since a1 is unsafe
        a1TruncatedMat = a1Truncated.toTensor(['A', 'B', 'C', 'D'])
        diff = np.linalg.norm(aMat - a1TruncatedMat)
        self.assertTrue(np.abs(diff) < 1e-10)

        a1 = a.copy()
        decomp = tensorSVDDecomposition(a1, rows = ['A', 'C'], cols = ['B', 'D'], chi = 16, keepdim = True, preserveLegs = True, keepLink = True)
        u, s, v = decomp['u'], decomp['s'], decomp['v']
        # print(u, s, v)

        # even after keepdim = True SVD, legs are still from a1
        for leg in a1.legs:
            self.assertNotEqual(leg.tensor, a1)
        
        for label in ['A', 'C']:
            self.assertEqual(a1.getLeg(label).tensor, u)
        for label in ['B', 'D']:
            self.assertEqual(a1.getLeg(label).tensor, v)
            
        self.assertIsNotNone(u.getLeg('inner:A|C').bond)
        self.assertIsNotNone(v.getLeg('inner:B|D').bond)
        for leg in s.legs:
            self.assertIsNotNone(leg.bond)

        a1Truncated = contractWithSequence([u, s, v])

        aMat = a.toTensor(['A', 'B', 'C', 'D']) # do not use a1, since a1 is unsafe
        a1TruncatedMat = a1Truncated.toTensor(['A', 'B', 'C', 'D'])
        diff = np.linalg.norm(aMat - a1TruncatedMat)
        self.assertTrue(np.abs(diff) < 1e-10)







