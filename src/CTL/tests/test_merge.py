from CTL.tests.packedTest import PackedTest
# from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.contract import  merge, shareBonds

import numpy as np 
# from ncon import ncon

class TestMerge(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Merge')

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