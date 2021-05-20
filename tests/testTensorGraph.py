from tests.packedTest import PackedTest
from CTL.tensor.contract.tensorGraph import TensorGraph
from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import generateOptimalSequence, makeTensorGraph

class TestTensorGraph(PackedTest):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'TensorGraph')

    def test_TensorGraph(self):
        a = Tensor(shape = (3, 4, 5), labels = ['a3', 'b4', 'c5'])
        b = Tensor(shape = (3, 6), labels = ['a3', 'd6'])
        c = Tensor(shape = (4, 6, 5), labels = ['b4', 'd6', 'c5'])

        makeLink(a.getLeg('a3'), b.getLeg('a3'))
        makeLink(a.getLeg('b4'), c.getLeg('b4'))
        makeLink(a.getLeg('c5'), c.getLeg('c5'))
        makeLink(b.getLeg('d6'), c.getLeg('d6'))

        tensorList = [a, b, c]

        tensorGraph = makeTensorGraph(tensorList)

        seq = tensorGraph.optimalContractSequence(tensorList, typicalDim = None)
        self.assertListEqual(seq, [(0, 2), (1, 0)])
        self.assertEqual(tensorGraph.optimalCostResult(), 378)
        # print(seq)

        seq = tensorGraph.optimalContractSequence(tensorList, typicalDim = 10)
        self.assertListEqual(seq, [(0, 2), (1, 0)])
        self.assertEqual(tensorGraph.optimalCostResult(), 10100)

