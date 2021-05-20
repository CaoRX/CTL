import CTL.funcs.funcs as funcs 
from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.contract import shareBonds, contractTensors
from CTL.tensor.contract.tensorGraph import TensorGraph
import numpy as np

def contractCost(ta, tb):
    bonds = shareBonds(ta, tb)
    intersectionShape = tuple([bond.legs[0].dim for bond in bonds])
    cost = funcs.tupleProduct(ta.shape) * funcs.tupleProduct(tb.shape) // funcs.tupleProduct(intersectionShape)
    costLevel = len(ta.shape) + len(tb.shape) - len(intersectionShape)
    return cost, costLevel

def makeTensorGraph(tensorList):
    # create a tensor graph based on the bonds in tensor list
    # UndirectedGraph is used
    # addFreeEdge for empty legs
    n = len(tensorList)
    g = TensorGraph(n)
    bondSet = set()
    idxDict = dict()
    for i in range(n):
        idxDict[tensorList[i]] = i
    for i in range(n):
        for leg in tensorList[i].legs:
            if (leg.bond is None):
                g.addFreeEdge(i, leg.dim)
            else:
                bond = leg.bond 
                if (bond in bondSet):
                    continue 
                bondSet.add(bond) 
                g.addEdge(i, idxDict[leg.anotherSide().tensor], leg.dim)
    
    g.addEdgeIndex()
    return g

def generateOptimalSequence(tensorList, bf = False, typicalDim = 10):
    tensorGraph = makeTensorGraph(tensorList)
    return tensorGraph.optimalContractSequence(bf = bf, typicalDim = typicalDim)

def contractWithSequence(tensorList, seq = None, bf = False, typicalDim = 10):
    if (seq is None):
        seq = generateOptimalSequence(tensorList, bf = bf, typicalDim = typicalDim)
    totalCost = 0.0
    totalLevel = 0

    for s, t in seq:
        cost, costLevel = contractCost(tensorList[s], tensorList[t])
        totalCost += cost 
        totalLevel = max(totalLevel, costLevel)
        tensorList[min(s, t)] = contractTensors(tensorList[s], tensorList[t])

    return tensorList[0]




