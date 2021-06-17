# a simple template of graphs
# G = G(V, E)
# each V contains several edges
# each edge contains two vertices

class GraphEdge:

    def __init__(self, a, b, weight = None):
        self.vertices = (a, b)
        self.weight = weight

class GraphVertex:

    def __init__(self, index, name = None):
        self.name = name 
        self.index = index 
        self.edges = []

    def addEdge(self, v, weight = None):
        newEdge = GraphEdge(self, v, weight = weight)
        self.edges.append(newEdge)
        return newEdge

    @property 
    def degree(self):
        return len(self.edges)

class Graph:

    def __init__(self, n):
        self.v = [GraphVertex(i) for i in range(n)]
        # self.e = []
        self.n = n 
        # self.m = 0

    def assertIndex(self, idx):
        assert ((idx >= 0) and (idx < self.n)), "Error: index {} is out of range of vertex indices [0, {}).".format(idx, self.n)

    def addEdge(self, idx1, idx2, weight = None):
        self.assertIndex(idx1)
        self.assertIndex(idx2)
        self.v[idx1].addEdge(self.v[idx2], weight = weight)

    # @property 
    # def edges(self):
    #     res = []
    #     for v in self.v:
    #         res += v.edges 
    #     return res
    
class UndirectedGraph(Graph):
    def __init__(self, n):
        super().__init__(n)
        self.edges = None
        self.updated = False
    
    def addEdge(self, idx1, idx2, weight = None):
        self.assertIndex(idx1)
        self.assertIndex(idx2)
        newEdge = self.v[idx1].addEdge(self.v[idx2], weight = weight)
        self.v[idx2].edges.append(newEdge)

        self.updated = True

    # def addFreeEdge(self, idx, weight = None):
    #     self.v[idx].addEdge(None, weight = weight)

    # @property 
    def getEdges(self):
        if (self.edges is not None) and (not self.updated):
            return self.edges
        res = []
        for v in self.v:
            for edge in v.edges:
                if (edge not in res):
                    res.append(edge) 
        # return list(set(res))
        self.edges = res 
        self.updated = False
        return res
    
    

