from CTL.funcs.graph import UndirectedGraph
import CTL.funcs.funcs as funcs

def squareLatticePBC(n, m = None, weight = 0.0):
    funcName = 'CTL.funcs.graphFuncs.squareLatticePBC'
    assert isinstance(n, int), funcs.errorMessage('n must be int, {} obtained.'.format(n), location = funcName)

    if funcs.isNumber(weight):
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
    funcName = 'CTL.funcs.graphFuncs.squareLatticeFBC'
    assert isinstance(n, int), funcs.errorMessage('n must be int, {} obtained.'.format(n), location = funcName)

    if funcs.isNumber(weight):
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
