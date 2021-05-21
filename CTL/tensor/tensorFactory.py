from CTL.tensor.tensor import Tensor 

def makeTriangleTensor(data, labels = ['1', '2', '3']):
    assert (len(data.shape) == 3), "Error: makeTriangleTensor can only accept tensor with 3 dimensions, but shape {} obtained.".format(data.shape)
    return Tensor(data = data, labels = labels)

def makeSquareTensor(data, labels = ['u', 'l', 'd', 'r']):
    assert (len(data.shape) == 4), "Error: makeSquareTensor can only accept tensor with 4 dimensions, but shape {} obtained.".format(data.shape)
    return Tensor(data = data, labels = labels)

def makeSquareOutTensor(data, loc):
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