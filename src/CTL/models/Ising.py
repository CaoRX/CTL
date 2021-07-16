# make tensors for Ising model

from CTL.tensor.tensor import Tensor
from CTL.tensor.diagonalTensor import DiagonalTensor
import CTL.funcs.funcs as funcs
from CTL.funcs.graph import UndirectedGraph
from CTL.tensor.contract.link import makeLink
import numpy as np 

def squareTensorMeasure(idx, obs = None):
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
    # tensor for square Ising model
    # use the simplest way to build on plaquettes, and linked with domain wall
    data = np.zeros((2, 2, 2, 2), dtype = np.float64)
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
        data[idx] = squareTensorMeasure(idx, obs) * np.exp(-beta * localE)
        data[idx] *= np.exp(symmetryBroken * squareTensorMeasure(idx, 'M'))
        # else:
        #     data[idx] = 0.0
    # print(data)

    return Tensor(labels = ['u', 'l', 'd', 'r'], data = data, degreeOfFreedom = 2)

def infiniteIsingExactM(T, V = 1.0):
    criticalT = 2.0 / np.log(1.0 + np.sqrt(2))
    if (T > criticalT):
        return 0.0
    else:
        return (1.0 - np.sinh(2 * V / T) ** (-4)) ** (0.125)

def IsingEdgeMatrix(betaJ):
    '''
    edge matrix of Ising model
    when combined with diagonal tensor at sites: tensor network of Ising model
    '''
    diag = np.sqrt(np.cosh(betaJ) * 0.5) + np.sqrt(np.sinh(betaJ) * 0.5)
    offDiag = np.sqrt(np.cosh(betaJ) * 0.5) - np.sqrt(np.sinh(betaJ) * 0.5)
    return np.array([[diag, offDiag], [offDiag, diag]])

def IsingSiteTensor(betaJ, dim = 4, labels = None):
    assert (funcs.isNumber(betaJ) or (len(betaJ) == dim)), funcs.errorMessage("betaJ {} do not have required dim {}.".format(betaJ, dim))
    assert ((labels is None) or (len(labels) == dim)), funcs.errorMessage("labels {} do not have required dim {}.".format(labels, dim))

    a = np.array([1.0, 1.0])
    a = funcs.diagonalMatrix(a, dim = dim)
    if (funcs.isNumber(betaJ)):
        betaJ = [betaJ] * dim
    # edgeMat = IsingEdgeMatrix(betaJ)
    for i in range(dim):
        edgeMat = IsingEdgeMatrix(betaJ[i])
        a = np.tensordot(a, edgeMat, (0, 0))
        # print(a)
    return Tensor(data = a, labels = labels)

def IsingTNFromUndirectedGraph(g):
    '''
    create a tensor network of Ising model, from an CTL.funcs.graph.UndirectedGraph
    the betaJ is set to be the weight on the edge
    '''

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
    return np.exp(-E)

def exactZFromGraphIsing(g):
    '''
    calculate the partition function Z Ising model, from an CTL.funcs.graph.UndirectedGraph
    the betaJ is set to be the weight on the edge
    '''

    funcName = 'CTL.models.Ising.exactZFromGraphIsing'
    assert (isinstance(g, UndirectedGraph)), funcs.errorMessage(err = "only UndirectedGraph can be trasferred to Ising tensor network, {} obtained.".format(g), location = funcName)

    res = 0.0
    n = len(g.v)
    # for S in range(1 << n):
    #     if (S % 10000 == 0):
    #         print('{}/{}'.format(S, 1 << n))
    #     res += getIsingWeight(g, S)
    res = np.sum(np.array([getIsingWeight(g, S) for S in range(1 << n)]))

    return res