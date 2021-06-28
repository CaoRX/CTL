import CTL.funcs.funcs as funcs 
from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.contract import shareBonds, contractTensors
from CTL.tensor.contract.tensorGraph import TensorGraph
from CTL.tensor.contract.link import makeLink
import numpy as np

def copyTensorList(tensorList):
    resTensorList = []
    tensorMap = dict()
    for tensor in tensorList:
        resTensorList.append(tensor.copy())
        tensorMap[tensor] = resTensorList[-1]
    # use the objects themselves as key, so no worry about double name
    
    addedBonds = set()
    for tensor in tensorList:
        for leg, newLeg1 in zip(tensor.legs, tensorMap[tensor].legs):
            if (leg.bond is not None) and (leg.bond not in addedBonds):
                addedBonds.add(leg.bond)
                leg2 = leg.anotherSide()

                newTensorB = tensorMap[leg2.tensor]

                newLeg2 = newTensorB.legs[leg2.tensor.legs.index(leg2)]
                makeLink(newLeg1, newLeg2)

                # no consideration about leg name, only from their relative positions
    
    return resTensorList 

def contractCost(ta, tb):
    diagonalA, diagonalB = ta.diagonalFlag, tb.diagonalFlag 
    if (diagonalA and diagonalB):
        return ta.bondDimension, 1
    elif (diagonalA):
        return tb.totalSize, tb.dim
    elif (diagonalB):
        return ta.totalSize, ta.dim 
    
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
    
    diagonalFlags = [tensor.diagonalFlag for tensor in tensorList]
    g = TensorGraph(n = n, diagonalFlags = diagonalFlags)
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

def contractAndCostWithSequence(tensorList, seq = None, bf = False, typicalDim = 10, inplace = False, outProductWarning = True):
    if (seq is None):
        seq = generateOptimalSequence(tensorList, bf = bf, typicalDim = typicalDim)
    totalCost = 0.0
    totalLevel = 0

    if (not inplace):
        tensorList = copyTensorList(tensorList)

    for s, t in seq:
        cost, costLevel = contractCost(tensorList[s], tensorList[t])
        totalCost += cost 
        totalLevel = max(totalLevel, costLevel)
        tensorList[min(s, t)] = contractTensors(tensorList[s], tensorList[t], outProductWarning = outProductWarning)

    return tensorList[0], totalCost

def contractWithSequence(tensorList, seq = None, bf = False, typicalDim = 10, inplace = False, outProductWarning = True):
    res, _ = contractAndCostWithSequence(tensorList = tensorList, seq = seq, bf = bf, typicalDim = typicalDim, inplace = inplace, outProductWarning = outProductWarning)
    return res

def contractTensorList(tensorList, outProductWarning = True):
    return contractWithSequence(tensorList, outProductWarning = outProductWarning)




