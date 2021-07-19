# a simple template of graphs
# G = G(V, E)
# each V contains several edges
# each edge contains two vertices

import CTL.funcs.funcs as funcs

class GraphEdge:
    """
    Edge object of the graphs.

    Parameters
    ----------
    a, b : any
        Two sides of the edge.
    weight : any, optional
        The weight on the edge.
    
    Attributes
    ----------
    vertices : tuple of any
        The two sides of the edge.
    weight : any
        The weight of the edge.
    """

    def __init__(self, a, b, weight = None):
        self.vertices = (a, b)
        self.weight = weight

    def anotherSide(self, a):
        """
        Another side of one vertex.

        Parameters
        ----------
        a : any
            One of the two vertices of this edge.
        
        Returns
        -------
        any
            The other side of this edge against a.
        """
        if (a == self.vertices[0]):
            return self.vertices[1]
        elif (a == self.vertices[1]):
            return self.vertices[0]
        else:
            raise ValueError(funcs.errorMessage(err = 'Graph edge does not contain vertex {}.'.format(a), location = 'GraphEdge.anotherSide'))

class GraphVertex:
    """
    Vertex object of the graphs.

    Parameters
    ----------
    index : int
        The index of the vertex in its graph.
    name : str, optional
        The name of the vertex.
    
    Attributes
    ----------
    name : str
        The name of the vertex.
    index : int
        The index of the vertex in its graph.
    edges : list of GraphEdge
        The edges start from this vertex.
    """

    def __init__(self, index, name = None):
        self.name = name 
        self.index = index 
        self.edges = []

    def __str__(self):
        return 'GraphVertex(index = {}, name = {})'.format(self.index, self.name)

    def addEdge(self, v, weight = None):
        """
        Add an edge starting on this vertex.

        Parameters
        ----------
        v : any
            The other side of the edge.
        weight : any, optional
            The weight of the added edge.
        
        Returns
        -------
        GraphEdge
            The edge object added.
        """
        newEdge = GraphEdge(self, v, weight = weight)
        self.edges.append(newEdge)
        return newEdge

    @property 
    def degree(self):
        """
        The out degree of the vertex

        Returns
        -------
        int
            The degree of current vertex.
        """
        return len(self.edges)

class Graph:
    """
    Object representing a graph G(V, E).

    Parameters
    ----------
    n : int
        The number of vertices in this graph.

    Attributes
    ----------
    v : list of GraphVertex
        The vertices of this graph.
    n : int
        The number of graphs in this graph.
    """
    def __init__(self, n):
        self.v = [GraphVertex(i) for i in range(n)]
        # self.e = []
        self.n = n 
        # self.m = 0

    def assertIndex(self, idx):
        """
        Decide whether an index representing one vertex in this graph.

        Parameters
        ----------
        idx : int
            The index required for some vertex.

        Raises
        ------
        AssertionError
            Raised when idx is not in [0, n), which means it is not a valid index.
        """
        assert ((idx >= 0) and (idx < self.n)), "Error: index {} is out of range of vertex indices [0, {}).".format(idx, self.n)

    def addEdge(self, idx1, idx2, weight = None):
        """
        Add an edge between two vertices.

        Parameters
        ----------
        idx1, idx2 : int
            Two indices of vertices that will be linked by the edge.
        weight : any, optional
            The weight of given edge.

        Returns 
        -------
        GraphEdge
            The edge added.
        """
        self.assertIndex(idx1)
        self.assertIndex(idx2)
        return self.v[idx1].addEdge(self.v[idx2], weight = weight)

    # @property 
    # def edges(self):
    #     res = []
    #     for v in self.v:
    #         res += v.edges 
    #     return res
    
class UndirectedGraph(Graph):
    """
    The object of undirected graph G(V, E), inheriting Graph.

    Parameters
    ----------
    n : int
        The number of vertices.
    
    Attributes
    ----------
    v : list of GraphVertex
        The vertices of this graph.
    n : int
        The number of graphs in this graph.
    edges : list of GraphEdge
        A snapshot of the set of edges, namely E. Note that it is E only when updated is False.
    updated : bool
        Whether the current undirected graph has been updated after the last time of obtaining edges.
    """
    def __init__(self, n):
        super().__init__(n)
        self.edges = None
        self.updated = False
    
    def addEdge(self, idx1, idx2, weight = None):
        """
        Add an undirected edge between two vertices.

        Parameters
        ----------
        idx1, idx2 : int
            Two indices of vertices that will be linked by the edge.
        weight : any, optional
            The weight of given edge.

        Returns 
        -------
        GraphEdge
            The edge added.
        """
        self.assertIndex(idx1)
        self.assertIndex(idx2)
        newEdge = self.v[idx1].addEdge(self.v[idx2], weight = weight)
        self.v[idx2].edges.append(newEdge)

        self.updated = True
        return newEdge

    # def addFreeEdge(self, idx, weight = None):
    #     self.v[idx].addEdge(None, weight = weight)

    # @property 
    def getEdges(self):
        """
        Get the edges of current graph. Note that it is not self.edges, since the addEdge function will not append the edge added to self.edges.

        After this function has been called, updated will be set to False, so that if we do not add edges again, the coming calls of this functio will automatically return the (cached) result. 

        Returns
        -------
        list of GraphEdge
            The E of the current graph.
        
        """
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
    
    

