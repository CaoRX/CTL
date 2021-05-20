import unittest 
from tests.packedTest import PackedTest

from CTL.funcs.graph import Graph, UndirectedGraph

class TestGraph(PackedTest):
    def test_graph(self):
        g = Graph(5)
        g.addEdge(1, 2)
        self.assertEqual(g.v[1].degree, 1)
        self.assertEqual(g.v[2].degree, 0)
    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'Graph')

class TestUndirectedGraph(PackedTest):
    def test_undirectedGraph(self):
        g = UndirectedGraph(5)
        g.addEdge(1, 2)
        self.assertEqual(g.v[1].degree, 1)
        self.assertEqual(g.v[2].degree, 1)

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName = methodName, name = 'UndirectedGraph')

    

if __name__ == '__main__':
    unittest.main()