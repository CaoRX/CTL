from CTL.tensor.bond import Bond, getBondName
from CTL.tensor.leg import Leg

import warnings
import CTL.funcs.funcs as funcs

def getLeg(tensor, leg):
    """
    Get the leg asked from a Tensor.

    Parameters
    ----------
    leg : str or Leg
    
    Returns
    -------
    Leg
        If leg itself is a Leg object, then it is fine. Otherwise, find the leg from the tensor with leg str as a label.
    """
    if (isinstance(leg, str)):
        assert (tensor is not None), "Error: ask leg {} by leg name but tensor is None in getLeg(tensor, leg).".format(leg)
        return tensor.getLeg(leg)
    else:
        return leg

def makeLink(legA, legB, tensorA = None, tensorB = None):
    """
    Make links between two legs.

    Parameters
    ----------
    legA, legB : str or Leg or list of Leg/str
        If str or Leg: then considered as 1-element list
        Each str represents a leg obtained from tensorA & tensorB
        Each leg can be taken as an independent object, without considering the Tensor.
    tensorA, tensorB : None or Tensor
        If None, then corresponding leg list must contain only Leg objects but not str.
        If Tensor, then the str's in leg lists can be transformed to Leg objects according the the corresponding Tensor.

    Returns
    -------
    bonds : list of Bond
        The bonds generated between the given legs.
    """
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
    """
    Merge the links between two tensors to make a larger bond

    Parameters
    ----------
    ta, tb : Tensor
        Two tensors, bonds between which will be merged.
    bondName : None or str
        If not None, then the new bond(and two legs) will be named with bondName. Otherwise, each side should be renamed as A|B|C|...|Z, where A, B, C... are the names of merged legs.
    renameWarning : bool
        If true, and only one bond is between ta and tb, then raise a warning since we do not merge any bonds but only rename the bond.

    Returns
    -------
    ta, tb : Tensor
        The two tensors with only one bond shared.
    """
    
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