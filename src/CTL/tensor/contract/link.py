from CTL.tensor.bond import Bond, getBondName
from CTL.tensor.leg import Leg

import warnings
import CTL.funcs.funcs as funcs

def getLeg(tensor, leg):
    if (isinstance(leg, str)):
        assert (tensor is not None), "Error: ask leg {} by leg name but tensor is None in getLeg(tensor, leg).".format(leg)
        return tensor.getLeg(leg)
    else:
        return leg

def makeLink(legA, legB, tensorA = None, tensorB = None):
    if (isinstance(legA, Leg) or isinstance(legA, str)):
        legA = [legA]
    if (isinstance(legB, Leg) or isinstance(legB, str)):
        legB = [legB]
    
    legA = [getLeg(tensorA, leg) for leg in legA]
    legB = [getLeg(tensorB, leg) for leg in legB]

    assert (len(legA) == len(legB)), "Error: the number of legs from A({}) and B({}) are not the same.".format(len(legA), len(legB))

    bonds = []
    for leg1, leg2 in zip(legA, legB):
        assert (leg1.dim == leg2.dim), "Error: leg {} and {} to be linked do not have the same dimensions.".format(leg1, leg2)
        bonds.append(Bond(leg1, leg2))
    
    return bonds

def mergeLink(ta, tb, bondName = None, renameWarning = True):
    '''
    merge the links between ta and tb
    if no links: warning
    if one link: warning(rename if bondName is not None)
    if two or more links: take the outProduct of each side, and merge the result
    '''
    
    funcName = 'CTL.tensor.contract.link.mergeLink'
    mergeLegA = []
    mergeLegB = []
    for leg in ta.legs:
        if (leg.bond is not None) and (leg.anotherSide().tensor == tb):
            mergeLegA.append(leg)
            mergeLegB.append(leg.anotherSide())

    if (len(mergeLegA) == 0):
        warnings.warn(funcs.warningMessage(warn = 'mergeLink cannot merge links between two tensors {} and {} sharing no bonds, do nothing'.format(ta, tb), location = funcName), RuntimeWarning)
        return ta, tb
    
    if (len(mergeLegA) == 1):
        if (renameWarning):
            warnings.warn(funcs.warningMessage(warn = 'mergeLink cannot merge links between two tensors {} and {} sharing one bond, only rename'.format(ta, tb), location = funcName), RuntimeWarning)
        if (bondName is not None):
            mergeLegA[0].name = bondName 
            mergeLegB[0].name = bondName
        return ta, tb

    # otherwise, we need to do outProduct, and then merge
    # bondNameA = funcs.combineName(namesList = [leg.name for leg in mergeLegA], givenName = bondName)
    # bondNameB = funcs.combineName(namesList = [leg.anem for leg in mergeLegB], givenName = bondName)
    if (bondName is None):
        bondNameListA = [leg.name for leg in mergeLegA]
        bondNameListB = [leg.name for leg in mergeLegB]
        bondNameA = '|'.join(bondNameListA)
        bondNameB = '|'.join(bondNameListB)
    elif (isinstance(bondName, str)):
        bondNameA = bondName 
        bondNameB = bondName 
    else:
        bondNameA, bondNameB = bondName # tuple/list

    newLegA = ta.outProduct(legList = mergeLegA, newLabel = bondNameA)
    newLegB = tb.outProduct(legList = mergeLegB, newLabel = bondNameB)

    makeLink(newLegA, newLegB)

    return ta, tb