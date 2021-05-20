from tests.packedTest import PackedTest
from CTL.tensornetwork.tensordict import TensorDict 
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor

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
        self.assertListEqual(FTN.optimalSeq, [(0, 1), (2, 0)])

        result2 = FTN.contract(tensorDict)
        self.assertEqual(int(result2.a), 36000)
        self.assertListEqual(FTN.optimalSeq, [(0, 1), (2, 0)])

        newShapeA = (3, 4, 5)
        newShapeB = (3, 6)

        newA = Tensor(shape = newShapeA, labels = ['a300', 'b4', 'c5'], data = np.ones(newShapeA))
        newB = Tensor(shape = newShapeB, labels = ['a300', 'd6'], data = np.ones(newShapeB))

        tensorDict.setTensor('a', newA) 
        tensorDict.setTensor('b', newB)

        result3 = FTN.contract(tensorDict) 
        self.assertEqual(int(result3.a), 360)
        self.assertListEqual(FTN.optimalSeq, [(0, 2), (1, 0)])
        self.assertEqual(FTN.bondDims['a-a300'], 3)
        self.assertEqual(FTN.tensorCount, 3)

        # problem: now the tensor network can only be contracted once
        # this can be solved by create another (copyable) FiniteTensorNetwork object
        # which traces all the bonds and legs, and can be easily copied