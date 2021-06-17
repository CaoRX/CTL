import unittest 

import CTL.funcs.funcs as funcs 
from CTL.funcs.stringSet import StringSet
from tests.packedTest import PackedTest

class TestTupleProduct(PackedTest):

    def test_tupleProduct(self):
        self.assertEqual(funcs.tupleProduct((2, 3)), 6)
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'tupleProduct')

class TestStringSet(PackedTest):
    def test_StringSet(self):
        ss = StringSet()
        ss.add('abc')
        ss.add('def')
        self.assertTrue(ss.contains('abc'))
        self.assertFalse(ss.contains('cde'))
        self.assertFalse(ss.newString() in ['abc', 'def'])
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'StringSet')

# if __name__ == '__main__':
#     unittest.main()