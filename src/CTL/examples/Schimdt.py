# do Schimdt decompostion over two tensors

from CTL.tensor.contract.contract import shareBonds, contractTwoTensors
import CTL.funcs.funcs as funcs
from CTL.tensor.tensor import Tensor
from CTL.tensor.leg import Leg
from CTL.tensor.diagonalTensor import DiagonalTensor
from CTL.tensor.contract.link import makeLink

def SchimdtDecomposition(ta, tb, chi):
    '''
    Schimdt decomposition between tensor ta and tb
    return ta, s, tb
    ta should be in canonical form, that is, a a^dagger = I
    to do this, first contract ta and tb, while keeping track of legs from a and legs from b
    then SVD over the matrix, take the required chi singular values
    take first chi eigenvectors for a and b, create a diagonal tensor for singular value tensor
    '''

    sb = shareBonds(ta, tb)
    assert (len(sb) > 0), funcs.errorMessage("Schimdt Decomposition cannot accept two tensors without common bonds, {} and {} gotten.".format(ta, tb), location = 'CTL.examples.Schimdt.SchimdtDecomposition')

    assert (ta.xp == tb.xp), funcs.errorMessage("Schimdt Decomposition cannot accetp two tensors with different xp: {} and {} gotten.".format(ta.xp, tb.xp), location = 'CTL.examples.Schimdt.SchimdtDecomposition')

    ta.addTensorTag('a')
    tb.addTensorTag('b') 

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

    mat = tot.toMatrix(rows = labelA, cols = labelB)

    np = ta.xp # default numpy

    u, s, vh = np.linalg.svd(mat)

    chi = min([chi, totShapeA, totShapeB])
    u = u[:, :chi]
    s = s[:chi]
    vh = vh[:chi]

    outLegForU = Leg(None, chi, name = 'o')
    inLegForU = Leg(None, chi, name = 'l')
    inLegForV = Leg(None, chi, name = 'r')
    outLegForV = Leg(None, chi, name = 'o')

    uTensor = Tensor(data = u, legs = legA + [outLegForU], shape = shapeA + (chi, ))
    sTensor = DiagonalTensor(data = s, legs = [inLegForU, inLegForV], shape = (chi, chi))
    vTensor = Tensor(data = vh, legs = [outLegForV] + legB, shape = (chi, ) + shapeB)

    # legs should be automatically set by Tensor / DiagonalTensor, so no need for setTensor
    
    # outLegForU.setTensor(uTensor)
    # outLegForV.setTensor(vTensor)

    # inLegForU.setTensor(sTensor)
    # inLegForV.setTensor(sTensor)

    # remove a- and b-
    for leg in uTensor.legs:
        if (leg.name.startswith('a-')):
            leg.name = leg.name[2:]

    for leg in vTensor.legs:
        if (leg.name.startswith('b-')):
            leg.name = leg.name[2:]

    makeLink(outLegForU, inLegForU)
    makeLink(outLegForV, inLegForV)

    return uTensor, sTensor, vTensor

    
    
