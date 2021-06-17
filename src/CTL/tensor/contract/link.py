from CTL.tensor.bond import Bond, getBondName
from CTL.tensor.leg import Leg

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