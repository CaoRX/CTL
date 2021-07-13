import unittest 

class PackedTest(unittest.TestCase):

    # def setUp(self):
    #     print('')
    #     if (self.name is not None):
    #         print('Begin test {}.'.format(self.name))

    # def tearDown(self):
    #     if (self.name is not None):
    #         print('End test {}.'.format(self.name))

    def showTestCaseBegin(self, name):
        print("\nBegin test {}.".format(name))
    def showTestCaseEnd(self, name):
        print("End test {}.".format(name))

    def __init__(self, methodName = 'runTest', name = None):
        super().__init__(methodName)
        self.name = name