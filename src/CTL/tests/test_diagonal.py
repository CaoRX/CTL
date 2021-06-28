from CTL.tests.packedTest import PackedTest
from CTL.tensor.diagonalTensor import DiagonalTensor

from CTL.tensor.tensor import Tensor 
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractTensorList, generateOptimalSequence
from CTL.tensor.contract.contract import contractTensors
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
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a', 'b', 'c'])
        b = Tensor(shape = (2, ), labels = ['x'])
        c = Tensor(shape = (2, ), labels = ['y'])

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)
        seq = generateOptimalSequence([a, b, c], typicalDim = None)
        # print('optimal sequence = {}'.format(seq))
        prod, cost = contractAndCostWithSequence([a, b, c], seq = seq)
        # prod = contractTensorList([a, b, c], outProductWarning = False)
        self.assertTrue(funcs.compareLists(prod.labels, ['c']))
        self.assertListEqual(seq, [(0, 2), (1, 0)])
        self.assertEqual(cost, 6.0)

        # if we use Tensor instead of DiagonalTensor for a
        # then the cost should be 12.0, and the order should be (1, 2), (0, 1)
        # the optimal cost of diagonal tensors can be achieved if we use diagonal nature for contraction
