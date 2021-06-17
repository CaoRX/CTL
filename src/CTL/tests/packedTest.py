import unittest 

class PackedTest(unittest.TestCase):

    def setUp(self):
        print('')
        if (self.name is not None):
            print('Begin test {}.'.format(self.name))

    def tearDown(self):
        if (self.name is not None):
            print('End test {}.'.format(self.name))

    def __init__(self, methodName = 'runTest', name = None):
        super().__init__(methodName)
        self.name = name