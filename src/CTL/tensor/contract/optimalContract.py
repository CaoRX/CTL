import CTL.funcs.funcs as funcs 
from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.contract import shareBonds, contractTwoTensors
from CTL.tensor.contract.tensorGraph import TensorGraph
from CTL.tensor.contract.link import makeLink
import numpy as np

def copyTensorList(tensorList, tensorLikeFlag = False, linkOutgoingBonds = False):
    """
    Make a copy of tensor list. Usually used for not-in-place tensor contraction.

    The difference from [x.copy() for x in tensorList]: it will copy all the bonds between tensors in tensorList.

    Parameters
    ----------
    tensorList : list of Tensor

    tensorLikeFlag : bool, default False
        Whether we are copying a set of TensorLikes(instead of tensors).
    linkOutgoingBonds : bool, default False
        Whether we connect the bonds outgoing(not in tensorList). If true, then all the old bonds will be replaced with new bonds generated for the copies. This is not recommended behavior since the function changes something it should not touch, but this is saved for future usage.
    
    Returns
    -------
    resTensroList : list of Tensor
        A copy of the given tensor list.
    """
    resTensorList = []
    tensorMap = dict()
    for tensor in tensorList:
        if (tensorLikeFlag):
            resTensorList.append(tensor.toTensorLike())
        else:
            resTensorList.append(tensor.copy())
        tensorMap[tensor] = resTensorList[-1]
    # use the objects themselves as key, so no worry about double name
    # print(tensorMap)
    
    addedBonds = set()
    for tensor in tensorList:
        for leg, newLeg1 in zip(tensor.legs, tensorMap[tensor].legs):
            if (leg.bond is not None) and (leg.bond not in addedBonds):
                leg2 = leg.anotherSide()

                addedBonds.add(leg.bond)

                if (leg2.tensor not in tensorList):
                    if (linkOutgoingBonds):
                        del leg.bond 
                        makeLink(newLeg1, leg2)
                    continue

                newTensorB = tensorMap[leg2.tensor]

                newLeg2 = newTensorB.legs[leg2.tensor.legs.index(leg2)]
                makeLink(newLeg1, newLeg2)

                # no consideration about leg name, only from their relative positions
    
    return resTensorList 

def contractCost(ta, tb):
    """
    The cost of contraction of two tensors.

    Parameters
    ----------
    ta, tb : Tensor
        Two tensors we want to contract.

    Returns
    -------
    cost : int
        The exact cost for contraction of two tensors(e.g. for two matrices A * B & B * C, the cost is A * B * C. However, for diagonal tensors, the cost is just the size of output tensor).
    costLevel : int
        The order of the cost(how many dimensions). This is used when we want to decide the order of our calculation to a given bond dimension chi.
    """
    diagonalA, diagonalB = ta.diagonalFlag, tb.diagonalFlag 
    if (diagonalA and diagonalB):
        return ta.bondDimension(), 1

    diagonal = diagonalA or diagonalB
    
    bonds = shareBonds(ta, tb)
    intersectionShape = tuple([bond.legs[0].dim for bond in bonds])
    if (not diagonal):
        cost = funcs.tupleProduct(ta.shape) * funcs.tupleProduct(tb.shape) // funcs.tupleProduct(intersectionShape)
        costLevel = len(ta.shape) + len(tb.shape) - len(intersectionShape)
    else:
        cost = funcs.tupleProduct(ta.shape) * funcs.tupleProduct(tb.shape) // (funcs.tupleProduct(intersectionShape) ** 2)
        costLevel = len(ta.shape) + len(tb.shape) - 2 * len(intersectionShape)
    return cost, costLevel

def makeTensorGraph(tensorList):
    """
    Make a tensor graph according to a list of tensor.

    Parameters
    ----------
    tensorList : list of Tensor

    Returns
    -------
    TensorGraph
        A tensor graph containing all the input tensors and bonds between them. For details, check CTL.tensor.contract.tensorGraph.
    """
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
    """
    Generate the optimal contraction sequence for a list of tensors.

    Parameters
    ----------
    tensorList : list of Tensor

    bf : bool
        Whether to search the order brute-force, or with ncon techniques.
    typicalDim : int or None
        If int, then in the calculation of cost, all bonds are supposed to be the same dimension as typicalDim.
        If None, then calculated with the real dimension.
        TODO: Make an option that cost are sorted with the polynomials, which is just the same when typicalDim = inf.

    Returns
    -------
    list of length-2 tuple of ints
        Length of the returned list should be $n - 1$, where $n$ is the length of tensorList.
        Every element contains two integers a < b, means we contract a-th and b-th tensors in tensorList, and save the new tensor into a-th location.
    """
    tensorGraph = makeTensorGraph(tensorList)
    return tensorGraph.optimalContractSequence(bf = bf, typicalDim = typicalDim)

def generateGreedySequence(tensorList):
    """
    Generate a good order of contraction, with a greedy algorithm of choosing the smallest size of the output tensor.

    Used for CATN or other cases where we cannot generate the optimal sequence since the number of tensors is too large.

    Parameters
    ----------
    tensorList : list of Tensor

    Returns
    -------
    list of length-2 tuple of ints
        Length of the returned list should be $n - 1$, where $n$ is the length of tensorList.
        Every element contains two integers a < b, means we contract a-th and b-th tensors in tensorList, and save the new tensor into a-th location.
    """
    tensorGraph = makeTensorGraph(tensorList)
    return tensorGraph.optimalContractSequence(greedy = True, typicalDim = None)

def contractAndCostWithSequence(tensorList, seq = None, bf = False, typicalDim = 10, inplace = False, outProductWarning = True, greedy = False):
    """
    The main function for contraction of tensor lists.

    Given a list of tensor and some options, make a contraction.

    Supporting input sequence and auto-generated sequence for contraction.

    Parameters
    ----------
    tensorList : list of Tensor

    seq : None or list of length-2 tuple of ints
        The sequence to be used for contraction, generated with generateGreedySequence or generateOptimalSequence. If None, then auto-generated here.
    bf : bool
        Whether to generate a sequence with brute-force algorithm.
    typicalDim : int
        If int, then in the calculation of cost, all bonds are supposed to be the same dimension as typicalDim.
        If None, then calculated with the real dimension.
    inplace : bool, default False
        Whether to contract inplace(so the original tensors will be destroyed). True can give more efficiency while False can keep the original data.
    outProductWarning : bool, default True
        Whether to raise a warning when finding outer product during contraction.
        In some cases, making outer product first can make a better order.
        However, outer product may also happen when we want to contract two set of tensors without bonds between them, usually comes from a mistake. So if you are not going to do this, turn the flag True can help debugging.
    greedy : bool, default False
        Whether to generate a sequence with greedy algorithm. If True, then the sequence may not be optimal. This is for large tensor networks where the optimal sequence is very time comsuming.

    Returns
    -------
    Tensor
        The result tensor of contraction.
    totalCost : int
        The exact cost for this contraction process.
    """
    if (seq is None):
        # print('{} tensors'.format(len(tensorList)))
        if (greedy):
            seq = generateGreedySequence(tensorList)
        else:
            seq = generateOptimalSequence(tensorList, bf = bf, typicalDim = typicalDim)
        # print(seq)
    totalCost = 0.0
    totalLevel = 0

    if (not inplace):
        tensorList = copyTensorList(tensorList)

    for s, t in seq:
        cost, costLevel = contractCost(tensorList[s], tensorList[t])
        totalCost += cost 
        totalLevel = max(totalLevel, costLevel)
        tensorList[min(s, t)] = contractTwoTensors(tensorList[s], tensorList[t], outProductWarning = outProductWarning)

    return tensorList[0], totalCost

def contractWithSequence(tensorList, seq = None, bf = False, typicalDim = 10, inplace = False, outProductWarning = True):
    """
    For more information, check contractAndCostWithSequence. This function is a single functional usage for contraction of tensor lists to a single tensor.

    Returns
    -------
    Tensor
        The result tensor of contraction.
    """
    res, _ = contractAndCostWithSequence(tensorList = tensorList, seq = seq, bf = bf, typicalDim = typicalDim, inplace = inplace, outProductWarning = outProductWarning)
    return res

def contractTensorList(tensorList, outProductWarning = True):
    """
    For more information, check contractAndCostWithSequence. This function is a simple demo for tensor contraction, where most of the options have been set automatically.

    Parameters
    ----------
    tensorList : list of Tensor

    outProductWarning : bool, default True

    Returns
    -------
    Tensor
        The resulf tensor of contraction.
    """
    return contractWithSequence(tensorList, outProductWarning = outProductWarning)




