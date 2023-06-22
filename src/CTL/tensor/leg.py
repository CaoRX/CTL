class Leg:
    """
    The leg class for tensors to connect.

    Parameters
    ----------
    tensor : None or Tensor
        The tensor where this leg locates at. It can be given as None and set later.
    dim : int
        The number of indices this leg have.
    name : str, optional
        The name of this leg. Will also show as the labels for the Tensor.
    
    Attributes
    ----------
    tensor : None or Tensor

    name : None or str

    dim : int

    bond : None or Bond
        If None, then the leg is free.
        If Bond, then the leg is linked to another leg of a tensor, and the bond is the link between.
    """

    def __init__(self, tensor, dim, name = None):
        self.tensor = tensor 
        self.name = name
        self.dim = dim
        self.bond = None
        # legs can have the same name: nothing will be affected
    
    def anotherSide(self):
        """
        Find the other side of the link on this leg.

        Returns
        -------
        None or Leg
            If the leg is free, then None. Otherwise the other side of the bond.
        """
        if (self.bond is None):
            return None 
        else:
            return self.bond.anotherSide(self)

    def setTensor(self, tensor):
        """
        Set the tensor for the leg.

        Parameters
        ----------
        tensor : Tensor
            The tensor that the leg locates at.
        """
        self.tensor = tensor

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
            bondStr = '[free]'
        
        return 'leg(dim = {}{}{}){}'.format(self.dim, nameStr, parentStr, bondStr)