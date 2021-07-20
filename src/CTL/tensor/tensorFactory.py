from CTL.tensor.tensor import Tensor 

def makeTriangleTensor(data, labels = ['1', '2', '3']):
    """
    Make a tensor of 3 legs.

    Parameters
    ----------
    data : 3-D ndarray of float
        Three-dimensional tensor that will be considered as the data for the tensros.
    labels : list of str, default ['1', '2', '3']
        Labels for three legs.
    
    Returns
    -------
    Tensor
        A tensor of 3 legs.
    """
    assert (len(data.shape) == 3), "Error: makeTriangleTensor can only accept tensor with 3 dimensions, but shape {} obtained.".format(data.shape)
    return Tensor(data = data, labels = labels)

def makeSquareTensor(data, labels = ['u', 'l', 'd', 'r']):
    """
    Make a tensor of 4 legs.

    Parameters
    ----------
    data : 4-D ndarray of float
        Four-dimensional tensor that will be considered as the data for the tensros.
    labels : list of str, default ['u', 'l', 'd', 'r']
        Labels for four legs.
    
    Returns
    -------
    Tensor
        A tensor of 4 legs.
    """
    assert (len(data.shape) == 4), "Error: makeSquareTensor can only accept tensor with 4 dimensions, but shape {} obtained.".format(data.shape)
    return Tensor(data = data, labels = labels)

def makeSquareOutTensor(data, loc):
    """
    Make an out-going tensor on one corner of a square.
    
    Parameters
    ----------
    data : 3-D ndarray of float
        The data for the out-going tensor.
    loc : {"ul", "ur", "dl", "dr"}
        One of the four locations on a square.

    Returns
    -------
    Tensor
        3-D tensor with legs named according to its location(loc), and the outer leg is named "o".
    """
    labels = []
    assert (loc in ['ul', 'ur', 'dl', 'dr']), "Error: loc of makeSquareOutTensor must be one of {}, but {} gotten.".format(['ul', 'ur', 'dl', 'dr'], loc)

    if (loc == 'ul'):
        labels = ['d', 'r', 'o']
    if (loc == 'ur'):
        labels = ['d', 'l', 'o']
    if (loc == 'dl'):
        labels = ['u', 'r', 'o']
    if (loc == 'dr'):
        labels = ['u', 'l', 'o']

    return Tensor(data = data, labels = labels)