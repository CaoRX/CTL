# make tensors for Ising model
import CTL.funcs.xplib as xplib

from CTL.tensor.tensor import Tensor
from CTL.tensor.diagonalTensor import DiagonalTensor
import CTL.funcs.funcs as funcs
from CTL.funcs.graph import UndirectedGraph
from CTL.tensor.contract.link import makeLink
# import numpy as np 

def squareTensorMeasure(idx, obs = None):
    """
    Local measurement on a square of four Ising spins.

    Parameters
    ----------
    idx : (4, ) tuple of {0, 1}
        The local spins on the square.
    obs : str, {'M', 'E'}, optional
        The observable to be measured. If None, then return 1.0.

    Returns
    -------
    float
        If obs is 'M': return the local magnetization(since each site is shared by two squares if we divide the system into such squares, only 0.5 counted for one spin).
        If obs is 'E': return the local energy(if neighbor spin is equal, then -1, else +1)
        If obs is None: considering partition function, return 1.0
    """
    if (obs is None):
        return 1.0 
    funcs.assertInSet(obs, ['M', 'E'], 'Ising obervables')
    if (obs == 'M'):
        res = 0.0
        for x in idx:
            if (x == 0):
                res -= 0.5 
            else:
                res += 0.5
        
        return res

    if (obs == 'E'):
        res = 0.0
        for i in range(4):
            if (idx[i] == idx[(i + 1) % 4]):
                res -= 1.0
            else:
                res += 1.0
        return res
    return 1.0

def squareIsingTensor(beta, obs = None, symmetryBroken = 0.0):
    """
    Tensor for square lattice Ising model, based on square decomposition.

    Parameters
    ----------
    beta : float
        The inverse temperature(suppose J = 1).
    obs : str, {'M', 'E'}, optional
        The observable to be measured. If None, then measuring Z.
    symmetryBroken : float, default 0.0
        A small value corresponding to the symmetry broken, used for calculating magnetization.

    Returns
    -------
    Tensor
        A tensor with four legs corresponding to four spins, and contain local weight or measurement(depending on obs).
    """
    # tensor for square Ising model
    # use the simplest way to build on plaquettes, and linked with domain wall
    data = xplib.xp.zeros((2, 2, 2, 2), dtype = xplib.xp.float64)
    for s in range(16):
        idx = funcs.intToBitTuple(s, 4)
        localE = 0.0
        # idxSum = 0
        # if 0, then localE += beta * 0.5
        # otherwise, localE -= beta * 0.5
        # sum must be even

        for i in range(4):
            # idxSum += idx[i]
            if (idx[i] == idx[(i + 1) % 4]):
                localE -= 1.0
            else:
                localE += 1.0
        
        # if (idxSum % 2 == 0):
        data[idx] = squareTensorMeasure(idx, obs) * xplib.xp.exp(-beta * localE)
        data[idx] *= xplib.xp.exp(symmetryBroken * squareTensorMeasure(idx, 'M'))
        # else:
        #     data[idx] = 0.0
    # print(data)

    return Tensor(labels = ['u', 'l', 'd', 'r'], data = data, degreeOfFreedom = 2)

def plaquetteIsingTensor(weight, diamondForm = False):
    """
    A local tensor of Ising model, based on plaquette. Including the interactions on four edges.

    Parameters
    ----------
    weight : float or length-2 tuple of float
        The weights(J) of Ising model. If a length-2 tuple, then two values represent (J_vertical, J_horizontal)
    diamondForm : bool, default False
        If True, then instead of usual ['lu', 'ru', 'rd', 'ld'] tensor, the tensor will be rotated 45 degrees clockwise, so that the ['lu', 'ru', 'rd', 'ld'] will be ['u', 'r', 'd', 'l'], so vertical weight will become the weight from left-bottom to right-top

    Returns
    -------
    Tensor
        Plaquette tensor of labels ['lu', 'ru', 'rd', 'ld'] or ['l', 'd', 'r', 'u']
    """

    funcName = 'CTL.models.Ising.plaquetteIsingTensor'
    if isinstance(weight, float):
        weight = [weight, weight]
    else:
        assert len(weight) == 2, funcs.errorMessage('Only float or (float, float) is accepted by {}, {} obtained.'.format(funcName, weight), location = funcName)
        weight = list(weight)

    data = xplib.xp.zeros((2, 2, 2, 2), dtype = xplib.xp.float64)
    # ru, rd, ld, lu
    for s in range(16):
        idx = funcs.intToBitTuple(s, 4)
        localE = 0.0

        for i in range(4):
            if (idx[i] == idx[(i + 1) % 4]):
                localE -= weight[i % 2]
                # e.g. ru & rd will share a bond of weight[0](J_vertical)
            else:
                localE += weight[i % 2]
        
        data[idx] = xplib.xp.exp(-localE)

    labels = ['ru', 'rd', 'ld', 'lu']
    if diamondForm:
        labels = ['r', 'd', 'l', 'u']
    return Tensor(labels = labels, data = data, degreeOfFreedom = 2)

def infiniteIsingExactM(T, V = 1.0):
    """
    Exact Magnetization of Ising model to the thermodynamical limit.

    Parameters
    ----------
    T : float
        The temperature.
    V : float, default 1.0
        The interaction in Ising model. Only V / T matters.
    
    Returns
    -------
    float
        The exact M at temperature T, when interaction is V. For T higher than T_c, it will return 0, otherwise from the exact solution of square lattice Ising model.
    """
    criticalT = 2.0 / xplib.xp.log(1.0 + xplib.xp.sqrt(2))
    if (T / V > criticalT):
        return 0.0
    else:
        return (1.0 - xplib.xp.sinh(2 * V / T) ** (-4)) ** (0.125)

def IsingEdgeMatrix(betaJ):
    """
    The edge tensor for site-based tensor network of Ising model.

    Parameters
    ----------
    betaJ : float
        The interaction combined with inverse temperature, namely, J / T.
    
    Returns
    -------
    ndarray of (2, 2)
        The local edge tensor for one site (diagonal) tensor to absorb.
    """
    diag = xplib.xp.sqrt(xplib.xp.cosh(betaJ) * 0.5) + xplib.xp.sqrt(xplib.xp.sinh(betaJ) * 0.5)
    offDiag = xplib.xp.sqrt(xplib.xp.cosh(betaJ) * 0.5) - xplib.xp.sqrt(xplib.xp.sinh(betaJ) * 0.5)
    return xplib.xp.array([[diag, offDiag], [offDiag, diag]])

def IsingSiteTensor(betaJ, dim = 4, labels = None):
    """
    The site tensor that can be connected to form Ising tensor network.

    Parameters
    ----------
    betaJ : float or list of float
        The interaction combined with inverse temperature, namely, J / T. When a list is given, the length should be dim, and each for one different edge.
    dim : int, default 4
        The degree of sites(the number of edges it is linked to). By default, square lattice value 4.
    labels : list of str, optional
        The labels of the result tensor on each leg. If betaJ is a number, since the legs are the same, the order of labels do not matter. Otherwise please be carefully about the order of labels since it corresponds to the order of betaJ.
    
    Returns
    -------
    Tensor
        The tensor of dim legs, labelled with labels, and representing the local interaction around a site(a diagonal site tensor with multiple edge tensors).
    """
    assert (funcs.isRealNumber(betaJ) or (len(betaJ) == dim)), funcs.errorMessage("betaJ {} do not have required dim {}.".format(betaJ, dim))
    assert ((labels is None) or (len(labels) == dim)), funcs.errorMessage("labels {} do not have required dim {}.".format(labels, dim))

    a = xplib.xp.array([1.0, 1.0])
    a = funcs.diagonalNDTensor(a, dim = dim)
    if (funcs.isRealNumber(betaJ)):
        betaJ = [betaJ] * dim
    # edgeMat = IsingEdgeMatrix(betaJ)
    for i in range(dim):
        edgeMat = IsingEdgeMatrix(betaJ[i])
        a = xplib.xp.tensordot(a, edgeMat, (0, 0))
        # print(a)
    return Tensor(data = a, labels = labels)

def IsingTNFromUndirectedGraph(g):
    """
    Create a tensor network of Ising model basing on an undirected graph.

    Parameters
    ----------
    g : UndirectedGraph
        The site-graph to add interaction. The weights represent the betaJ on each edge.
    
    Returns
    -------
    list of Tensor
        A tensor network, each of the tensors represents one site, and the contraction of this tensor network will give the exact partition function Z.
    """

    funcName = 'CTL.models.Ising.IsingTNFromUndirectedGraph'
    assert (isinstance(g, UndirectedGraph)), funcs.errorMessage(err = "only UndirectedGraph can be trasferred to Ising tensor network, {} obtained.".format(g), location = funcName)

    nodes = g.v
    edgeIndex = dict()
    for ei, edge in enumerate(g.getEdges()):
        edgeIndex[edge] = ei
    
    def getLegName(edge, toV):
        return str(toV.index) + '-' + str(edgeIndex[edge])
    def getLabels(v):
        return [getLegName(edge = e, toV = e.anotherSide(v)) for e in v.edges]
    def getWeights(v):
        return [e.weight for e in v.edges]

    tensors = [IsingSiteTensor(betaJ = getWeights(v), dim = len(v.edges), labels = getLabels(v)) for v in nodes]

    for ei, edge in enumerate(g.getEdges()):
        v1, v2 = edge.vertices 
        idx1, idx2 = v1.index, v2.index 
        makeLink(getLegName(edge = edge, toV = v2), getLegName(edge = edge, toV = v1), tensors[idx1], tensors[idx2])

    return tensors

def getIsingWeight(g, S):
    """
    Calculate the weight of Ising model for a graph and given spins.

    Parameters
    ----------
    g : UndirectedGraph
        The site-graph to add interaction. The weights represent the betaJ on each edge.
    S : int
        The bitmask of the Ising spin states.
    
    Returns
    -------
    float
        The Boltzmann weight for this configuration.

    """
    funcName = 'CTL.models.Ising.getIsingWeight'
    assert (isinstance(g, UndirectedGraph)), funcs.errorMessage(err = "only UndirectedGraph can be trasferred to Ising tensor network, {} obtained.".format(g), location = funcName)

    E = 0.0
    n = len(g.v)
    spin = funcs.intToBitList(S, n)
    for edge in g.getEdges():
        v1, v2 = edge.vertices 
        s1, s2 = spin[v1.index], spin[v2.index]
        if (s1 == s2):
            E -= edge.weight 
        else:
            E += edge.weight 
    return xplib.xp.exp(-E)

def exactZFromGraphIsing(g):
    """
    Calculate the exact partition function by enumerating configurations of Ising model.

    Parameters
    ----------
    g : UndirectedGraph
        The site-graph to add interaction. The weights represent the betaJ on each edge.

    Returns
    -------
    float
        The exact Z.
    """

    funcName = 'CTL.models.Ising.exactZFromGraphIsing'
    assert (isinstance(g, UndirectedGraph)), funcs.errorMessage(err = "only UndirectedGraph can be trasferred to Ising tensor network, {} obtained.".format(g), location = funcName)

    res = 0.0
    n = len(g.v)
    # for S in range(1 << n):
    #     if (S % 10000 == 0):
    #         print('{}/{}'.format(S, 1 << n))
    #     res += getIsingWeight(g, S)
    res = xplib.xp.sum(xplib.xp.array([getIsingWeight(g, S) for S in range(1 << n)]))

    return res