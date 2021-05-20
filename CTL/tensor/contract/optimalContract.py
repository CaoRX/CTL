import CTL.funcs.funcs as funcs 
from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.contract import shareBonds, contractTensors
from CTL.tensor.contract.tensorGraph import TensorGraph
import numpy as np

def contractCost(ta, tb):
    bonds = shareBonds(ta, tb)
    intersectionShape = tuple([bond.leg1.dim for bond in bonds])
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
        for leg in len(tensorList[i].legs):
            if (leg.bond is None):
                g.addFreeEdge(i, leg.dim)
            else:
                bond = leg.bond 
                if (bond in bondSet):
                    continue 
                bondSet.add(bond) 
                g.addEdge(i, idxDict[leg.anotherSide.tensor], leg.dim)
    
    return g

def contractWithSequence(tensorList, seq = None, bf = True):
    if (seq is None):
        tensorGraph = makeTensorGraph(tensorList)
        seq = tensorGraph.optimalContractSequence(bf = bf)
    totalCost = 0.0
    totalLevel = 0

    for s, t in seq:
        cost, costLevel = contractCost(tensorList[s], tensorList[t])
        totalCost += cost 
        totalLevel = max(totalLevel, costLevel)
        tensorList[min(s, t)] = contractTensors(tensorList[s], tensorList[t])

    return tensorList[0]




