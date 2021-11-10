from CTL.funcs.stringSet import StringSet, getBondName
from CTL.tensor.leg import Leg
from CTL.funcs.funcs import errorMessage

class Bond:
    """
    The bond object linking two legs.

    Parameters
    ----------
    leg1, leg2 : Leg
        Two legs to be connected.
    name : str, optional
        The name of the bond.

    Attributes
    ----------
    name : str

    legs : tuple of Leg
        Two legs that has been connected.
    """
    # bondNameSet = set([])
    # bondNameSet = StringSet()


    def __init__(self, leg1, leg2, name = None):
        assert (isinstance(leg1, Leg) and (isinstance(leg2, Leg))), errorMessage(err = "Bond must be initialized with 2 Leg elements.", location = 'Bond.__init__')
        assert (leg1.dim == leg2.dim), errorMessage(err = "{} and {} do not share the same dimension.".format(leg1, leg2), location = 'Bond.__init__')
        # self.name = getBondName(name)
        self.name = name
        self.legs = (leg1, leg2)
        leg1.bond = self 
        leg2.bond = self

    def __repr__(self):
        return "Bond(name = {}, leg1 = {}, leg2 = {})".format(self.name, self.legs[0], self.legs[1])

    def anotherSide(self, leg):
        """
        Given one side of leg, return the other side.

        Parameters
        ----------
        leg : Leg
            One side of the bond.

        Returns
        -------
        Leg
            The other side of the given leg.
        
        Raises
        ------
        AssertionError
            Raised if the given leg is not one of the two legs of this bond.
        """
        leg1, leg2 = self.legs 
        if (leg1 == leg):
            return leg2
        elif (leg2 == leg):
            return leg1 
        else:
            assert False, "Error: {} is not in ({}, {}).".format(leg, leg1, leg2)
            return None

    def sideLeg(self, tensor):
        """
        Given tensor on one side, return the corresponding leg. If both legs are from that tensor, return the first.

        Parameters
        ----------
        tensor : Tensor
            Tensor on one side.

        Returns
        -------
        Leg
            The leg in the given tensor.
        
        Raises
        ------
        AssertionError
            Raised if the both legs are not in given tensor.
        """
        if (self.legs[0] in tensor.legs):
            return self.legs[0]
        elif (self.legs[1] in tensor.legs):
            return self.legs[1]
        else:
            raise ValueError(errorMessage("legs {} is not in tensor {}.".format(self.legs, tensor), location = 'Bond.sideLeg'))

# getBondName = Bond.bondNameSet.newString