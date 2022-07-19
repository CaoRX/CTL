from CTL.tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor
import CTL.funcs.funcs as funcs

import numpy as np 

class TestFTN(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'TensorFTN')

    def test_FTN(self):
        shapeA = (300, 4, 5)
        shapeB = (300, 6)
        shapeC = (4, 6, 5)
        a = Tensor(shape = shapeA, labels = ['a300', 'b4', 'c5'], data = np.ones(shapeA))
        b = Tensor(shape = shapeB, labels = ['a300', 'd6'], data = np.ones(shapeB))
        c = Tensor(shape = shapeC, labels = ['b4', 'd6', 'c5'], data = np.ones(shapeC))

        tensorDict = TensorDict()
        tensorDict.setTensor('a', a) 
        tensorDict.setTensor('b', b)
        tensorDict.setTensor('c', c)

        FTN = FiniteTensorNetwork(['a', 'b'], realCost = True)
        FTN.addTensor('c')

        FTN.addLink('a', 'a300', 'b', 'a300')
        FTN.addLink('a', 'b4', 'c', 'b4')
        FTN.addLink('a', 'c5', 'c', 'c5')
        FTN.addLink('b', 'd6', 'c', 'd6')

        result = FTN.contract(tensorDict)
        self.assertEqual(int(result.a), 36000)
        self.assertListEqual(FTN.optimalSeq, [(0, 1), (0, 2)])

        result2 = FTN.contract(tensorDict)
        self.assertEqual(int(result2.a), 36000)
        self.assertListEqual(FTN.optimalSeq, [(0, 1), (0, 2)])

        newShapeA = (3, 4, 5)
        newShapeB = (3, 6)

        newA = Tensor(shape = newShapeA, labels = ['a300', 'b4', 'c5'], data = np.ones(newShapeA))
        newB = Tensor(shape = newShapeB, labels = ['a300', 'd6'], data = np.ones(newShapeB))

        tensorDict.setTensor('a', newA) 
        tensorDict.setTensor('b', newB)

        result3 = FTN.contract(tensorDict) 
        self.assertEqual(int(result3.a), 360)
        self.assertListEqual(FTN.optimalSeq, [(0, 2), (0, 1)])
        self.assertEqual(FTN.bondDims['a-a300'], 3)
        self.assertEqual(FTN.tensorCount, 3)

    def test_FreeBondFTN(self):
        shapeA = (300, 4, 5)
        shapeB = (300, 6)
        shapeC = (4, 6, 5)
        a = Tensor(shape = shapeA, labels = ['a300', 'b4', 'c5'], data = np.ones(shapeA))
        b = Tensor(shape = shapeB, labels = ['a300', 'd6'], data = np.ones(shapeB))
        c = Tensor(shape = shapeC, labels = ['e4', 'd6', 'c5'], data = np.ones(shapeC))

        tensorDict = TensorDict()
        tensorDict.setTensor('a', a) 
        tensorDict.setTensor('b', b)
        tensorDict.setTensor('c', c)

        FTN = FiniteTensorNetwork(['a', 'b'], realCost = True)
        FTN.addTensor('c')

        FTN.addLink('a', 'a300', 'b', 'a300')
        FTN.addLink('a', 'c5', 'c', 'c5')
        FTN.addLink('b', 'd6', 'c', 'd6')

        result = FTN.contract(tensorDict, removeTensorTag = False)
        self.assertTrue(funcs.compareLists(result.labels, ['a-b4', 'c-e4']))
        self.assertEqual(int(result.a[0][1]), 9000)

        result = FTN.contract(tensorDict, removeTensorTag = True)
        self.assertTrue(funcs.compareLists(result.labels, ['b4', 'e4']))
        self.assertEqual(int(result.a[0][1]), 9000)

        FTN.unlock()
        FTN.addPostNameChange('c', 'e4', 'e4FromC')
        FTN.addPreNameChange('a', 'b4', 'b4FromA')
        FTN.addPreNameChange('a', 'a300', 'a3')
        FTN.removePreNameChange('a', 'a300', 'a3')
        FTN.addPostNameChange('a', 'd6', 'foo')
        FTN.removePostNameChange('a', 'd6', 'foo')

        result = FTN.contract(tensorDict, removeTensorTag = True)
        # print(result.labels)
        self.assertTrue(funcs.compareLists(result.labels, ['b4FromA', 'e4FromC']))
        self.assertEqual(int(result.a[0][1]), 9000)

        FTN.unlock()
        FTN.removePostNameChange('c', 'e4', 'e4FromC')
        FTN.addPreNameChange('c', 'e4', 'e4FromC')
        FTN.addPostOutProduct([('a', 'b4FromA'), ('c', 'e4FromC')], 'out')
        result = FTN.contract(tensorDict, removeTensorTag = True)

        self.assertListEqual(result.labels, ['out'])
        self.assertEqual(result.shape[0], 16)

    def test_FTNPreConjugate(self):
        # TODO: add tests for preConjugate
        pass
