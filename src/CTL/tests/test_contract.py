from CTL.tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
# from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor
from CTL.tensor.diagonalTensor import DiagonalTensor
from CTL.tensor.tensorFactory import makeSquareTensor, makeTriangleTensor, makeSquareOutTensor
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.contract import contractTwoTensors, merge, shareBonds
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
        self.assertListEqual(seq, [(0, 2), (1, 0)])
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
        # print('seq = {}'.format(generateOptimalSequence([a, b, c])))

        a = Tensor(data = aData, labels = ['a', 'b', 'c'], tensorLikeFlag = True)
        b = Tensor(data = bData, labels = ['x'], tensorLikeFlag = True)
        c = Tensor(data = cData, labels = ['y'], tensorLikeFlag = True)

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)

        res2, cost2 = contractAndCostWithSequence([a, b, c])
        self._testIsTensorLike(res2)
        # self.assertListEqual(list(res1.a), list(res2.a))
        # self.assertTrue(funcs.floatArrayEqual(res1.a, res2.a))

        # print(cost1, cost2)

        # print(res1.a, res2.a)

        # test diagonal tensor contraction
        a = DiagonalTensor(shape = (2, 2), labels = ['a1', 'a2'], tensorLikeFlag = True)
        b = DiagonalTensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        res = contractTwoTensors(a, b)
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
            _ = contractTwoTensors(a, b)
        
        self.assertRaises(TypeError, contractBetweenTensorAndTensorLike)
    
    def test_extremeContraction(self):

        aData = np.random.random_sample(2)
        aTensorData = np.array([[[aData[0], 0], [0, 0]], [[0, 0], [0, aData[1]]]])
        bData = np.random.random_sample((2, 2))
        a = DiagonalTensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res1 = contractTwoTensors(a, b)
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res2 = contractTwoTensors(b, a)
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
        res1 = contractTwoTensors(a, b)
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2), labels = ['a1', 'a2'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2), labels = ['b1', 'b2'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a2', 'b2', a, b)
        res2 = contractTwoTensors(b, a)
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
        res1 = contractTwoTensors(a, b)
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        makeLink('a3', 'b2', a, b)
        res2 = contractTwoTensors(b, a)
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
        res1 = contractTwoTensors(a, b)
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 2, 2), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res2 = contractTwoTensors(b, a)
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
        res1 = contractTwoTensors(a, b)
        self._testIsTensorLike(res1)

        a = Tensor(shape = (2, 2, 2), labels = ['a1', 'a2', 'a3'], data = aTensorData, tensorLikeFlag = True)
        b = Tensor(shape = (2, 4, 7), labels = ['b1', 'b2', 'b3'], data = bData, tensorLikeFlag = True)
        makeLink('a1', 'b1', a, b)
        # makeLink('a3', 'b2', a, b)
        res2 = contractTwoTensors(b, a)
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

    def test_merge(self):
        '''
        test the merge(ta, tb) for merge the bonds between ta and tb
        '''

        # normal tensor merge case: 2 shared bonds
        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'])
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'])

        makeLink('a3', 'b3', ta, tb)
        makeLink('a4', 'b4', ta, tb)

        ta, tb = merge(ta, tb)
        
        self.assertTrue(funcs.compareLists(ta.labels, ['a3|a4', 'a5']))
        self.assertTrue(funcs.compareLists(tb.labels, ['b3|b4', 'b6']))
        ta.reArrange(['a3|a4', 'a5'])
        self.assertTupleEqual(ta.shape, (12, 5))
        tb.reArrange(['b3|b4', 'b6'])
        self.assertTupleEqual(tb.shape, (12, 6))
        self.assertEqual(len(shareBonds(ta, tb)), 1)

        # normal tensor merge case, order changed

        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'])
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'])

        makeLink('a3', 'b3', ta, tb)
        makeLink('a4', 'b4', ta, tb)

        tb, ta = merge(tb, ta)
        
        self.assertTrue(funcs.compareLists(ta.labels, ['a4|a3', 'a5']))
        self.assertTrue(funcs.compareLists(tb.labels, ['b4|b3', 'b6']))

        # test single shared bond: with warning, and do nothing but rename

        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'])
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'])

        makeLink('a3', 'b3', ta, tb)
        # tb, ta = merge(tb, ta, bondName = 'o')

        with self.assertWarns(RuntimeWarning) as cm:
            tb, ta = merge(tb, ta, bondName = 'o')
        
        self.assertIn('link.py', cm.filename)
        message = cm.warning.__str__()
        self.assertIn('mergeLink cannot merge links', message)
        self.assertIn('sharing one bond', message)
        self.assertTrue(funcs.compareLists(['o', 'a4', 'a5'], ta.labels))
        self.assertTrue(funcs.compareLists(['o', 'b4', 'b6'], tb.labels))

        # test for normal merge, tensorLike

        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'], tensorLikeFlag = True)
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'], tensorLikeFlag = True)

        makeLink('a3', 'b3', ta, tb)
        makeLink('a4', 'b4', ta, tb)

        ta, tb = merge(ta, tb)
        
        self.assertTrue(funcs.compareLists(ta.labels, ['a3|a4', 'a5']))
        self.assertTrue(funcs.compareLists(tb.labels, ['b3|b4', 'b6']))
        ta.reArrange(['a3|a4', 'a5'])
        self.assertTupleEqual(ta.shape, (12, 5))
        tb.reArrange(['b3|b4', 'b6'])
        self.assertTupleEqual(tb.shape, (12, 6))
        self.assertEqual(len(shareBonds(ta, tb)), 1)
        self.assertTrue(ta.tensorLikeFlag and tb.tensorLikeFlag)

        # test for single bond merge, tensorLike

        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'], tensorLikeFlag = True)
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'], tensorLikeFlag = True)

        makeLink('a3', 'b3', ta, tb)
        # tb, ta = merge(tb, ta, bondName = 'o')

        with self.assertWarns(RuntimeWarning) as cm:
            tb, ta = merge(tb, ta, bondName = 'o')
        
        self.assertIn('link.py', cm.filename)
        message = cm.warning.__str__()
        self.assertIn('mergeLink cannot merge links', message)
        self.assertIn('sharing one bond', message)
        self.assertTrue(ta.tensorLikeFlag and tb.tensorLikeFlag)

        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'])
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'])
        
        with self.assertWarns(RuntimeWarning) as cm:
            ta, tb = merge(ta, tb, bondName = 'o')
        self.assertIn('link.py', cm.filename)

        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'])
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'])

        with self.assertWarns(RuntimeWarning) as cm:
            ta, tb = merge(ta, tb, bondName = 'o', chi = 2)
        # print(cm.__dict__)
        self.assertIn('link.py', cm.filename)

        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'])
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'])

        makeLink('a3', 'b3', ta, tb)
        makeLink('a4', 'b4', ta, tb)

        ta, tb = merge(ta, tb, chi = 2)
        
        self.assertTrue(funcs.compareLists(ta.labels, ['a3|a4', 'a5']))
        self.assertTrue(funcs.compareLists(tb.labels, ['b3|b4', 'b6']))
        ta.reArrange(['a3|a4', 'a5'])
        self.assertTupleEqual(ta.shape, (2, 5))
        tb.reArrange(['b3|b4', 'b6'])
        self.assertTupleEqual(tb.shape, (2, 6))
        self.assertEqual(len(shareBonds(ta, tb)), 1)

        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'], tensorLikeFlag = True)
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'], tensorLikeFlag = True)

        makeLink('a3', 'b3', ta, tb)
        makeLink('a4', 'b4', ta, tb)

        ta, tb = merge(ta, tb, chi = 2)
        # print(ta, tb)
        
        self.assertTrue(funcs.compareLists(ta.labels, ['a3|a4', 'a5']))
        self.assertTrue(funcs.compareLists(tb.labels, ['b3|b4', 'b6']))
        ta.reArrange(['a3|a4', 'a5'])
        self.assertTupleEqual(ta.shape, (2, 5))
        tb.reArrange(['b3|b4', 'b6'])
        self.assertTupleEqual(tb.shape, (2, 6))
        self.assertEqual(len(shareBonds(ta, tb)), 1)
        self.assertTrue(ta.tensorLikeFlag and tb.tensorLikeFlag)

        ta = Tensor(shape = (3, 4, 5), labels = ['a3', 'a4', 'a5'])
        tb = Tensor(shape = (4, 3, 6), labels = ['b4', 'b3', 'b6'])

        makeLink('a3', 'b3', ta, tb)
        makeLink('a4', 'b4', ta, tb)

        # large chi test
        ta, tb = merge(ta, tb, chi = 15)
        # the real internal bond size is chosen by min(chi, ta.remainShape, tb.remainShape, mergeShape)
        
        self.assertTrue(funcs.compareLists(ta.labels, ['a3|a4', 'a5']))
        self.assertTrue(funcs.compareLists(tb.labels, ['b3|b4', 'b6']))
        ta.reArrange(['a3|a4', 'a5'])
        self.assertTupleEqual(ta.shape, (5, 5))
        tb.reArrange(['b3|b4', 'b6'])
        self.assertTupleEqual(tb.shape, (5, 6))
        self.assertEqual(len(shareBonds(ta, tb)), 1)


        




        

        

