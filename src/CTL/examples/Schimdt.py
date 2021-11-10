# do Schimdt decompostion over two tensors
import CTL.funcs.xplib as xplib
from CTL.tensor.contract.contract import shareBonds, contractTwoTensors
import CTL.funcs.funcs as funcs
from CTL.tensor.tensor import Tensor
from CTL.tensor.leg import Leg
from CTL.tensor.diagonalTensor import DiagonalTensor
from CTL.tensor.contract.link import makeLink
# import numpy as np


def SchimdtDecomposition(ta, tb, chi, squareRootSeparation = False, swapLabels = ([], []), singularValueEps = 1e-10):
    '''
    Schimdt decomposition between tensor ta and tb
    return ta, s, tb
    ta should be in canonical form, that is, a a^dagger = I
    to do this, first contract ta and tb, while keeping track of legs from a and legs from b
    then SVD over the matrix, take the required chi singular values
    take first chi eigenvectors for a and b, create a diagonal tensor for singular value tensor

    if squareRootSeparation is True: then divide s into two square root diagonal tensors
    and contract each into ta and tb, return ta, None, tb

    if swapLabels is not ([], []): swap the two set of labels for output, so we swapped the locations of two tensors on MPS
    e.g. t[i], t[i + 1] = SchimdtDecomposition(t[i], t[i + 1], chi = chi, squareRootSeparation = True, swapLabels = (['o'], ['o']))
    we can swap the two tensors t[i] & t[i + 1], both have an "o" leg connected to outside
    while other legs(e.g. internal legs in MPS, usually 'l' and 'r') will not be affected
    '''

    funcName = 'CTL.examples.Schimdt.SchimdtDecomposition'
    sb = shareBonds(ta, tb)
    assert (len(sb) > 0), funcs.errorMessage("Schimdt Decomposition cannot accept two tensors without common bonds, {} and {} gotten.".format(ta, tb), location = funcName)
    assert (ta.tensorLikeFlag == tb.tensorLikeFlag), funcs.errorMessage("Schimdt Decomposition must havge two objects being either Tensor or TensorLike simultaneously, but {} and {} obtained.".format(ta, tb), location = funcName)

    TLFlag = ta.tensorLikeFlag
    sbDim = funcs.tupleProduct(tuple([bond.legs[0].dim for bond in sb]))

    sharedLabelA = sb[0].sideLeg(ta).name
    sharedLabelB = sb[0].sideLeg(tb).name
    # if (sharedLabelA.startswith('a-')):
    #     raise ValueError(funcs.errorMessage(err = "shared label {} of tensor A starts with 'a-'.".format(sharedLabelA), location = funcName))
    # if (sharedLabelB.startswith('b-')):
    #     raise ValueError(funcs.errorMessage(err = "shared label {} of tensor B starts with 'b-'.".format(sharedLabelB), location = funcName))

    # assert (ta.xp == tb.xp), funcs.errorMessage("Schimdt Decomposition cannot accept two tensors with different xp: {} and {} gotten.".format(ta.xp, tb.xp), location = funcName)

    assert (len(swapLabels[0]) == len(swapLabels[1])), funcs.errorMessage(err = "invalid swap labels {}.".format(swapLabels), location = funcName)
    assert ta.labelsInTensor(swapLabels[0]), funcs.errorMessage(err = "{} not in tensor {}.".format(swapLabels[0], ta), location = funcName)
    assert tb.labelsInTensor(swapLabels[1]), funcs.errorMessage(err = "{} not in tensor {}.".format(swapLabels[1], tb), location = funcName)

    ta.addTensorTag('a')
    tb.addTensorTag('b') 
    for swapLabel in swapLabels[0]:
        ta.renameLabel('a-' + swapLabel, 'b-' + swapLabel)
    for swapLabel in swapLabels[1]:
        tb.renameLabel('b-' + swapLabel, 'a-' + swapLabel)

    tot = contractTwoTensors(ta, tb)

    legA = [leg for leg in tot.legs if leg.name.startswith('a-')]
    legB = [leg for leg in tot.legs if leg.name.startswith('b-')]

    labelA = [leg.name for leg in legA]
    labelB = [leg.name for leg in legB]
    # not remove a- and b- here, since we need to add an internal leg, and we need to distinguish it from others

    shapeA = tuple([leg.dim for leg in legA])
    shapeB = tuple([leg.dim for leg in legB])

    totShapeA = funcs.tupleProduct(shapeA)
    totShapeB = funcs.tupleProduct(shapeB)

    if (TLFlag):
        u = None 
        vh = None
        s = None
        chi = min([chi, totShapeA, totShapeB, sbDim])
    else:
        mat = tot.toMatrix(rows = labelA, cols = labelB)

        # np = ta.xp # default numpy

        u, s, vh = xplib.xp.linalg.svd(mat)

        chi = min([chi, totShapeA, totShapeB, funcs.nonZeroElementN(s, singularValueEps)])
        u = u[:, :chi]
        s = s[:chi]
        vh = vh[:chi]

    if (squareRootSeparation):
        if (TLFlag):
            uS = None 
            vS = None
        else:
            sqrtS = xplib.xp.sqrt(s)
            uS = funcs.rightDiagonalProduct(u, sqrtS)
            vS = funcs.leftDiagonalProduct(vh, sqrtS)

        outLegForU = Leg(None, chi, name = sharedLabelA)
        # inLegForU = Leg(None, chi, name = sharedLabelB)
        # internalLegForS1 = Leg(None, chi, name = 'o')
        # internalLegForS2 = Leg(None, chi, name = 'o')
        # inLegForV = Leg(None, chi, name = sharedLabelA)
        outLegForV = Leg(None, chi, name = sharedLabelB)

        uTensor = Tensor(data = uS, legs = legA + [outLegForU], shape = shapeA + (chi, ), tensorLikeFlag = TLFlag)
        # s1Tensor = DiagonalTensor(data = xplib.xp.sqrt(s), legs = [inLegForU, internalLegForS1], shape = (chi, chi))
        # s2Tensor = DiagonalTensor(data = xplib.xp.sqrt(s), legs = [internalLegForS2, inLegForV], shape = (chi, chi))
        vTensor = Tensor(data = vS, legs = [outLegForV] + legB, shape = (chi, ) + shapeB, tensorLikeFlag = TLFlag)

        # legs should be automatically set by Tensor / DiagonalTensor, so no need for setTensor
        
        # outLegForU.setTensor(uTensor)
        # outLegForV.setTensor(vTensor)

        # inLegForU.setTensor(sTensor)
        # inLegForV.setTensor(sTensor)

        # remove a- and b-
        for leg in legA:
            if (leg.name.startswith('a-')):
                leg.name = leg.name[2:]

        for leg in legB:
            if (leg.name.startswith('b-')):
                leg.name = leg.name[2:]

        makeLink(outLegForU, outLegForV)

        # makeLink(outLegForU, inLegForU)
        # makeLink(outLegForV, inLegForV)
        # makeLink(internalLegForS1, internalLegForS2)
        # uTensor = contractTwoTensors(uTensor, s1Tensor)
        # vTensor = contractTwoTensors(vTensor, s2Tensor)
        return uTensor, None, vTensor

    outLegForU = Leg(None, chi, name = sharedLabelA)
    inLegForU = Leg(None, chi, name = sharedLabelB)
    inLegForV = Leg(None, chi, name = sharedLabelA)
    outLegForV = Leg(None, chi, name = sharedLabelB)

    uTensor = Tensor(data = u, legs = legA + [outLegForU], shape = shapeA + (chi, ), tensorLikeFlag = TLFlag)
    sTensor = DiagonalTensor(data = s, legs = [inLegForU, inLegForV], shape = (chi, chi), tensorLikeFlag = TLFlag)
    vTensor = Tensor(data = vh, legs = [outLegForV] + legB, shape = (chi, ) + shapeB, tensorLikeFlag = TLFlag)

    # legs should be automatically set by Tensor / DiagonalTensor, so no need for setTensor
    
    # outLegForU.setTensor(uTensor)
    # outLegForV.setTensor(vTensor)

    # inLegForU.setTensor(sTensor)
    # inLegForV.setTensor(sTensor)

    # remove a- and b-
    for leg in legA:
        if (leg.name.startswith('a-')):
            leg.name = leg.name[2:]

    for leg in legB:
        if (leg.name.startswith('b-')):
            leg.name = leg.name[2:]

    makeLink(outLegForU, inLegForU)
    makeLink(outLegForV, inLegForV)

    return uTensor, sTensor, vTensor

def matrixSchimdtDecomposition(a, dim, chi = None):
    '''
    Schimdt decomposition of matrix a
    a can be either 1D or 2D
    1. if a is 1D: then we reshape a to (dim, -1)
    and then SVD to obtain u(dim, chi), s(chi, chi), vh(chi, -1), return u and s @ vh 
    2. if a is 2D: (s0, s1): then we reshape a to (s0 * dim, s1 // dim)
    SVD to obtain u(s0 * dim, chi), s(chi, chi), vh(chi, -1)
    return u(s0, dim, chi), s @ vh(chi, s1 // dim)
    '''

    funcName = 'CTL.examples.Schimdt.matrixSchimdtDecomposition'

    shape = a.shape

    assert (len(shape) in [1, 2]), funcs.errorMessage("matrix shape can only be 1D or 2D, {} obtained.".format(shape), location = funcName)

    if (len(shape) == 1):
        l = shape[0]
        assert ((dim != 0) and (l % dim == 0)), funcs.errorMessage("matrix shape {} cannot be Schimdt decomposed with dimension {}.".format(shape, dim), location = funcName)
        a = a.reshape((dim, -1))
        u, s, vh = xplib.xp.linalg.svd(a)
        if (chi is None):
            chi = min([u.shape[0], vh.shape[1], funcs.nonZeroElementN(s)])
        else:
            chi = min([u.shape[0], vh.shape[1], chi, funcs.nonZeroElementN(s)])
        
        u = u[:, :chi]
        s = s[:chi]
        vh = vh[:chi]

        return u, funcs.leftDiagonalProduct(vh, s)
    
    else:
        s0, s1 = shape
        assert ((dim != 0) and (s1 % dim == 0)), funcs.errorMessage("matrix shape {} cannot be Schimdt decomposed with dimension {}.".format(shape, dim), location = funcName)
        a = a.reshape(s0 * dim, -1)
        u, s, vh = xplib.xp.linalg.svd(a)
        if (chi is None):
            chi = min([u.shape[0], vh.shape[1], funcs.nonZeroElementN(s)])
        else:
            chi = min([u.shape[0], vh.shape[1], chi, funcs.nonZeroElementN(s)])

        u = u[:, :chi].reshape(s0, dim, chi)
        s = s[:chi]
        vh = vh[:chi]
        return u, funcs.leftDiagonalProduct(vh, s)
        
        

    
    
