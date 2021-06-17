class Leg:

    def __init__(self, tensor, dim, name = None):
        self.tensor = tensor 
        self.name = name
        self.dim = dim
        self.bond = None
        # legs can have the same name: nothing will be affected
    
    def anotherSide(self):
        if (self.bond is None):
            return None 
        else:
            return self.bond.anotherSide(self)

    def __repr__(self):
        # if (self.tensor.name is not None):
        parentStr = ', parent = {}'.format(self.tensor.__repr__())
        # else:
        # parentStr = ''
        
        if (self.name is not None):
            nameStr = ', name = {}'.format(self.name)
        else:
            nameStr = ''
        
        if (self.bond is not None):
            bondStr = ', bonded by Bond({})'.format(self.bond.name)
        else:
            bondStr = ''
        
        return 'leg(dim = {}{}{}){}'.format(self.dim, nameStr, parentStr, bondStr)