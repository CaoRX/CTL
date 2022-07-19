from CTL.tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
# from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor
from CTL.tensor.diagonalTensor import DiagonalTensor
from CTL.tensor.tensorFactory import makeSquareTensor, makeTriangleTensor, makeSquareOutTensor
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.contract import merge, shareBonds, contractTwoTensorsNotInPlace, contractTwoTensors
from CTL.tensor.contract.optimalContract import contractAndCostWithSequence, generateOptimalSequence, contractCost

import numpy as np 
# from ncon import ncon

class TestContract(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Contract')

    def _testIsTensorLike(self, tensor):
        self.assertTrue(tensor.tensorLikeFlag)
        self.assertIsNone(tensor.a)

    def test_contraction(self):
        print('Begin test diagonalTensor contraction.')
        a = DiagonalTensor(shape = (2, 2), labels = ['a', 'b'], tensorLikeFlag = True)
        b = Tensor(shape = (2, ), labels = ['x'], tensorLikeFlag = True)
        c = Tensor(shape = (2, ), labels = ['y'], tensorLikeFlag = True)

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)
        seq = generateOptimalSequence([a, b, c], typicalDim = 10)
        # print('optimal sequence = {}'.format(seq))
        prod, cost = contractAndCostWithSequence([a, b, c], seq = seq)
        # print('cost = {}'.format(cost))
        # prod = contractTensorList([a, b, c], outProductWarning = False)
        self._testIsTensorLike(prod)
        self.assertTrue(funcs.compareLists(prod.labels, []))
        self.assertListEqual(seq, [(0, 2), (0, 1)])
        self.assertEqual(cost, 4.0)

        # if we use Tensor instead of DiagonalTensor for a
        # then the cost should be 12.0, and the order should be (1, 2), (0, 1)
        # the optimal cost of diagonal tensors can be achieved if we use diagonal nature for contraction

        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a', 'b', 'c'], tensorLikeFlag = True)
        b = DiagonalTensor(shape = (2, 2), labels = ['x', 'y'], tensorLikeFlag = True)
        makeLink('a', 'x', a, b)
        prod, cost = contractAndCostWithSequence([a, b])
        self._testIsTensorLike(prod)
        self.assertEqual(cost, 2)
        self.assertTrue(funcs.compareLists(prod.labels, ['b', 'c', 'y']))

        aData = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 3]]])
        bData = np.random.random_sample(2)
        cData = np.random.random_sample(2)

        a = DiagonalTensor(data = aData, labels = ['a', 'b', 'c'], tensorLikeFlag = True)
        b = Tensor(data = bData, labels = ['x'], tensorLikeFlag = True)
        c = Tensor(data = cData, labels = ['y'], tensorLikeFlag = True)

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)

        res1, cost1 = contractAndCostWithSequence([a, b, c])
        self._testIsTensorLike(res1)
        self.assertTrue(funcs.floatEqual(cost1, 6.0))
        # print('seq = {}'.format(generateOptimalSequence([a, b, c])))

        a = Tensor(data = aData, labels = ['a', 'b', 'c'], tensorLikeFlag = True)
        b = Tensor(data = bData, labels = ['x'], tensorLikeFlag = True)
        c = Tensor(data = cData, labels = ['y'], tensorLikeFlag = True)

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)

        res2, cost2 = contractAndCostWithSequence([a, b, c])
        self._testIsTensorLike(res2)
        self.assertTrue(funcs.floatEqual(cost2, 12.0))
        # self.assertListEqual(list(res1.a), list(res2.a))
        # self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))

        # print(cost1, cost2)

        # print(res1.a, res2.a)

        # test diagonal tensor contraction
        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'], tensorLikeFlag = True)
        b = DiagonalTensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        res = a @ b
        self._testIsTensorLike(res)
        self.assertTupleEqual(res.shape, (2, 2, 2))
        self.assertEqual(res.dim, 3)
        self.assertTrue(res.diagonalFlag)
        # self.assertTrue((res.a == np.ones(2)).all())

        # test for diagonal * diagonal contraction cost(just O(length))
        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'], tensorLikeFlag = True)
        b = DiagonalTensor(shape = 2, labels = ['b1', 'b2'], tensorLikeFlag = True) # deduce dim
        makeLink('a1', 'b2', a, b)

        cost, _ = contractCost(a, b)
        self.assertEqual(cost, 2.0)

        res, cost = contractAndCostWithSequence([a, b])
        self._testIsTensorLike(res)
        self.assertEqual(res.dim, 2)
        self.assertEqual(res._length, 2)
        self.assertTupleEqual(res.shape, (2, 2))
        self.assertEqual(cost, 2.0)
        self.assertTrue(res.diagonalFlag)

        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'], tensorLikeFlag = True)
        b = Tensor(shape = (2, 3, 3), labels = ['b1', 'b2', 'b3'], tensorLikeFlag = True) # deduce dim
        makeLink('a1', 'b1', a, b)

        cost, _ = contractCost(a, b)

        self.assertEqual(cost, 18.0)

        res, cost = contractAndCostWithSequence([a, b])
        self._testIsTensorLike(res)
        # print(res)
        self.assertEqual(res.dim, 3)
        self.assertTrue(funcs.compareLists(list(res.shape), [2, 3, 3]))
        self.assertEqual(cost, 18.0)

        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'], tensorLikeFlag = True)
        b = Tensor(shape = (2, 3, 3), labels = ['b1', 'b2', 'b3'], tensorLikeFlag = False) # deduce dim
        makeLink('a1', 'b1', a, b)

        def contractBetweenTensorAndTensorLike():
            _ = a @ b
        
        self.assertRaises(TypeError, contractBetweenTensorAndTensorLike)
    
    def test_extremeContraction(self):

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 2))
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res1 = a @ b
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res2 = b @ a
        self._testIsTensorLike(res2)

        self.assertTrue(funcs.compareLists(list(res1.labels), ['a2']))
        self.assertTrue(funcs.compareLists(list(res2.labels), ['a2']))
        # self.assertListEqual(list(res1.a), list(res2.a))
        # self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))

        aData = np.random.random_sample(2)
        aTensorData = np.array([[aData[0], 0], [0, aData[1]]])
        bData = np.random.random_sample((2, 2))
        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'], data = aData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a2', 'b2', a, b)
        res1 = a @ b
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2), labels = ['a1', 'a2'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a2', 'b2', a, b)
        res2 = b @ a
        self._testIsTensorLike(res2)

        self.assertTrue(funcs.compareLists(res1.labels, []))
        self.assertTrue(funcs.compareLists(res2.labels, []))
        # self.assertListEqual(list(res1.single(), list(res2.a))
        # self.assertTrue(funcs.floatEqual(res1.single(), res2.single()))

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 2, 2))
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res1 = a @ b
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res2 = b @ a
        self._testIsTensorLike(res2)
        

        self.assertTrue(funcs.compareLists(list(res1.labels), ['a2', 'b3']))
        self.assertTrue(funcs.compareLists(list(res2.labels), ['a2', 'b3']))
        # print(res1.labels, res2.labels)
        # print(res1.a, res2.a)
        res2.reArrange(res1.labels)
        # self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))
        # self.assertListEqual(list(np.ravel(res1.a)), list(np.ravel(res2.a)))

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 2, 2))
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res1 = a @ b
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res2 = b @ a
        self._testIsTensorLike(res2)

        # print('ndEye(2, 2) = {}'.format(funcs.ndEye(2, 2)))
        # print('ndEye(1, 2) = {}'.format(funcs.ndEye(1, 2)))
        # print('ndEye(3, 2) = {}'.format(funcs.ndEye(3, 2)))

        self.assertTrue(funcs.compareLists(list(res1.labels), ['a2', 'a3', 'b2', 'b3']))
        self.assertTrue(funcs.compareLists(list(res2.labels), ['a2', 'a3', 'b2', 'b3']))
        # print(res1.labels, res2.labels)
        res2.reArrange(res1.labels)
        # print('res1 = {}, res2 = {}'.format(res1.a, res2.a))
        # self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))
        # self.assertListEqual(list(np.ravel(res1.a)), list(np.ravel(res2.a)))

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 4, 7))
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 4, 7), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res1 = a @ b
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 4, 7), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res2 = b @ a
        self._testIsTensorLike(res2)

        # print('ndEye(2, 2) = {}'.format(funcs.ndEye(2, 2)))
        # print('ndEye(1, 2) = {}'.format(funcs.ndEye(1, 2)))
        # print('ndEye(3, 2) = {}'.format(funcs.ndEye(3, 2)))

        self.assertTrue(funcs.compareLists(list(res1.labels), ['a2', 'a3', 'b2', 'b3']))
        self.assertTrue(funcs.compareLists(list(res2.labels), ['a2', 'a3', 'b2', 'b3']))
        # print(res1.labels, res2.labels)
        res2.reArrange(res1.labels)
        # print('res1 = {}, res2 = {}'.format(res1.a, res2.a))
        # self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))

    def test_contractTwoTensorsNotInPlace(self):
        """
        TODO: check the deprecated flag for diag-nondiag tensor contraction
        """

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 4, 7))
        cData = np.random.random_sample((4, 10))
        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = False)
        b = Tensor(shape = (2, 4, 7), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = False)
        c = Tensor(shape = (4, 10), labels = ['c1', 'c2'], data = cData, tensorLikeFlag = False)
        makeLink('a1', 'b1', a, b)
        makeLink('b2', 'c1', b, c)

        res1 = contractTwoTensors(a, b)
        # print(a, b, c, res1)
        self.assertEqual(len(shareBonds(c, res1)), 1)
        self.assertEqual(len(shareBonds(c, b)), 1)
        self.assertTrue(a.isDeprecated)
        self.assertTrue(b.isDeprecated)

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 4, 7))
        cData = np.random.random_sample((4, 10))
        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = False)
        b = Tensor(shape = (2, 4, 7), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = False)
        c = Tensor(shape = (4, 10), labels = ['c1', 'c2'], data = cData, tensorLikeFlag = False)
        makeLink('a1', 'b1', a, b)
        makeLink('b2', 'c1', b, c)

        res2 = contractTwoTensorsNotInPlace(a, b)
        # print(a, b, c, res2)
        
        self.assertEqual(len(shareBonds(c, res2)), 0)
        self.assertEqual(len(shareBonds(c, b)), 1)
        self.assertFalse(a.isDeprecated)
        self.assertFalse(b.isDeprecated)




        




        

        

