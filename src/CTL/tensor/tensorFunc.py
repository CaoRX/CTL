import CTL.funcs.funcs as funcs
def isIsometry(tensor, labels, eps = 1e-10):
    '''
    check whether the tensor is an isometry for given labels
    labels cannot contain all the labels in tensor
    '''

    assert (tensor.labelsInTensor(labels)), funcs.errorMessage("labels {} is not in {}.".format(labels, tensor), location = 'CTL.tensor.tensorFunc.isIsometry')
    mat = tensor.toMatrix(cols = labels)
    np = tensor.xp
    iden = mat @ funcs.transposeConjugate(mat, np = np)
    return funcs.checkIdentity(iden, eps = eps)
