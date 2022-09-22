import numpy as np
from CTL.tests.packedTest import PackedTest

class TestMPO(PackedTest):
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'MPO')

    