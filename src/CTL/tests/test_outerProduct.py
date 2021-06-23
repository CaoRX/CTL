from CTL.tests.packedTest import PackedTest

from CTL.tensor.tensor import Tensor 
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractTensorList
from CTL.tensor.contract.contract import contractTensors
import CTL.funcs.funcs as funcs

import numpy as np 

class TestOuterProduct(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'OuterProduct')

    def test_outerProduct(self):
        a = Tensor(shape = (2, ), labels = ['a'])
        b = Tensor(shape = (2, ), labels = ['b'])

        op = contractTensors(a, b, outProductWarning = False)
        self.assertTrue(funcs.compareLists(op.labels, ['a', 'b']))
        
        a = Tensor(shape = (2, 2, 2), labels = ['a', 'b', 'c'])
        b = Tensor(shape = (2, ), labels = ['x'])
        c = Tensor(shape = (2, ), labels = ['y'])

        makeLink('a', 'x', a, b)
        makeLink('b', 'y', a, c)
        prod = contractTensorList([a, b, c], outProductWarning = False)
        self.assertTrue(funcs.compareLists(prod.labels, ['c']))
        