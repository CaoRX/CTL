from CTL.funcs.stringSet import StringSet, getBondName
from CTL.tensor.leg import Leg

class Bond:
    # bondNameSet = set([])
    # bondNameSet = StringSet()

    def __init__(self, leg1, leg2, name = None):
        assert (isinstance(leg1, Leg) and (isinstance(leg2, Leg))), "Error: Bond must be initialized with 2 Leg elements."
        self.name = getBondName(name)
        self.legs = (leg1, leg2)
        leg1.bond = self 
        leg2.bond = self

    def __repr__(self):
        return "Bond(name = {}, leg1 = {}, leg2 = {})".format(self.name, self.legs[0], self.legs[1])

    def anotherSide(self, leg):
        leg1, leg2 = self.legs 
        if (leg1 == leg):
            return leg2
        elif (leg2 == leg):
            return leg1 
        else:
            assert False, "Error: {} is not in ({}, {}).".format(leg, leg1, leg2)
            return None

# getBondName = Bond.bondNameSet.newString