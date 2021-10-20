from CTL.funcs.graph import UndirectedGraph
import CTL.funcs.funcs as funcs

def squareLatticePBC(n, m = None, weight = 0.0):
    """
    Create a graph of square lattice, periodic boundary condition.

    Parameters
    ----------
    n : int
        The number of rows of the lattice.
    m : int, optional
        The number of columns of the lattice. By default, the same as n.
    weight : float or tuple of float, optional
        The weight of edges between vertices. If a tuple is given, then the first for vertical edges and the second for horizontal edges. By default, both weights are 0.0. This will be used in models like Ising model, where the weight can represent interactions.
    
    Returns
    -------
    UndirectedGraph
        A graph representing the lattice. 
    """
    funcName = 'CTL.funcs.graphFuncs.squareLatticePBC'
    assert isinstance(n, int), funcs.errorMessage('n must be int, {} obtained.'.format(n), location = funcName)

    if funcs.isRealNumber(weight):
        weight = (weight, weight)

    weightV, weightH = weight

    if (m is None):
        m = n 

    nn = n * m 

    def getIndex(x, y):
        x = funcs.safeMod(x, n)
        y = funcs.safeMod(y, m) 
        return x * m + y 

    g = UndirectedGraph(nn)
    for x in range(n):
        for y in range(m):
            g.addEdge(idx1 = getIndex(x, y), idx2 = getIndex(x, y + 1), weight = weightH)
            g.addEdge(idx1 = getIndex(x, y), idx2 = getIndex(x + 1, y), weight = weightV)
    
    return g 

def squareLatticeFBC(n, m = None, weight = 0.0):
    """
    Create a graph of square lattice, free boundary condition.

    Parameters
    ----------
    n : int
        The number of rows of the lattice.
    m : int, optional
        The number of columns of the lattice. By default, the same as n.
    weight : float or tuple of float, optional
        The weight of edges between vertices. If a tuple is given, then the first for vertical edges and the second for horizontal edges. By default, both weights are 0.0. This will be used in models like Ising model, where the weight can represent interactions.
    
    Returns
    -------
    UndirectedGraph
        A graph representing the lattice. 
    """
    funcName = 'CTL.funcs.graphFuncs.squareLatticeFBC'
    assert isinstance(n, int), funcs.errorMessage('n must be int, {} obtained.'.format(n), location = funcName)

    if funcs.isRealNumber(weight):
        weight = (weight, weight)

    weightV, weightH = weight

    if (m is None):
        m = n 

    nn = n * m 

    def getIndex(x, y):
        assert (x >= 0 and x < n and y >= 0 and y < m), funcs.errorMessage('({}, {}) is not a valid coordinate in ({}, {}) lattice.'.format(x, y, n, m), location = funcName + ".getIndex")
        return x * m + y 

    g = UndirectedGraph(nn)
    for x in range(n):
        for y in range(m):
            if (y < m - 1):
                g.addEdge(idx1 = getIndex(x, y), idx2 = getIndex(x, y + 1), weight = weightH)
            if (x < n - 1):
                g.addEdge(idx1 = getIndex(x, y), idx2 = getIndex(x + 1, y), weight = weightV)
    
    return g 

def doubleSquareLatticeFBC(n, m = None, weight = 0.0):
    """
    Create a graph of double square lattice, free boundary condition.

    Double square lattice: take the plaquette tensor as checker board, and they form a square lattice. The (n, m) lattice contains n(m + 1) + m(n + 1) sites.

    Parameters
    ----------
    n : int
        The number of rows of the lattice.
    m : int, optional
        The number of columns of the lattice. By default, the same as n.
    weight : float or tuple of float, optional
        The weight of edges between vertices. If a tuple is given, then the first for left-up to right-bottom edges and the second for right-up to left-bottom edges. By default, both weights are 0.0. This will be used in models like Ising model, where the weight can represent interactions.
    
    Returns
    -------
    UndirectedGraph
        A graph representing the lattice. 
    """
    funcName = 'CTL.funcs.graphFuncs.doubleSquareLatticeFBC'
    assert isinstance(n, int), funcs.errorMessage('n must be int, {} obtained.'.format(n), location = funcName)

    if funcs.isRealNumber(weight):
        weight = (weight, weight)

    weightUTB, weightBTU = weight

    if (m is None):
        m = n 

    nn = n * (m + 1) + m * (n + 1)

    def getIndex(d, x, y):
        # d == 0: "up/bottom" of plaquettes, (n + 1) * m
        # d == 1: "left/right" of plaquettes, n * (m + 1)
        # order: up to down, left to right
        assert (d != 0) or (x >= 0 and x <= n and y >= 0 and y < m), funcs.errorMessage('({}, {}) is not a valid coordinate in ({}, {}) lattice.'.format(x, y, n, m), location = funcName + ".getIndex")
        assert (d != 1) or (x >= 0 and x < n and y >= 0 and y <= m), funcs.errorMessage('({}, {}) is not a valid coordinate in ({}, {}) lattice.'.format(x, y, n, m), location = funcName + ".getIndex")
        
        if (d == 0):
            return x * m + y
        else:
            return (n + 1) * m + x * (m + 1) + y

    g = UndirectedGraph(nn)
    for x in range(n):
        for y in range(m):
            # add edges for square (x, y)
            g.addEdge(idx1 = getIndex(1, x, y), idx2 = getIndex(0, x, y), weight = weightBTU)
            g.addEdge(idx1 = getIndex(1, x, y), idx2 = getIndex(0, x + 1, y), weight = weightUTB)
            g.addEdge(idx1 = getIndex(1, x, y + 1), idx2 = getIndex(0, x, y), weight = weightUTB)
            g.addEdge(idx1 = getIndex(1, x, y + 1), idx2 = getIndex(0, x + 1, y), weight = weightBTU)
            # if (y < m - 1):
            #     g.addEdge(idx1 = getIndex(x, y), idx2 = getIndex(x, y + 1), weight = weightH)
            # if (x < n - 1):
            #     g.addEdge(idx1 = getIndex(x, y), idx2 = getIndex(x + 1, y), weight = weightV)
    
    return g 

def completeGraph(n, weight = 0.0):
    """
    Create a complete graph.

    Parameters
    ----------
    n : int
        The number of vertices.
    weight : float, optional
        The weight of edges between vertices. By default, both weights are 0.0. This will be used in models like Ising model, where the weight can represent interactions.
    
    Returns
    -------
    UndirectedGraph
        A complete graph of n vertices. 
    """
    funcName = 'CTL.funcs.graphFuncs.completeGraph'
    assert isinstance(n, int), funcs.errorMessage('n must be int, {} obtained.'.format(n), location = funcName)

    g = UndirectedGraph(n)
    for x in range(n):
        for y in range(x + 1, n):
            g.addEdge(idx1 = x, idx2 = y, weight = weight)
    return g