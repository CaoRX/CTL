import CTL.funcs.funcs as funcs
import warnings
def isIsometry(tensor, labels, eps = 1e-10):
    '''
    check whether the tensor is an isometry for given labels
    labels cannot contain all the labels in tensor
    '''
    funcName = 'CTL.tensor.tensorFunc.isIsometry'
    assert (tensor.labelsInTensor(labels)), funcs.errorMessage("labels {} is not in {}.".format(labels, tensor), location = funcName)
    if (tensor.tensorLikeFlag):
        warnings.warn(funcs.warningMessage(warn = 'isIsometry cannot work on tensorLike object {}, return True by default.'.format(tensor), location = funcName))
        return True
    mat = tensor.toMatrix(cols = labels)
    np = tensor.xp
    iden = mat @ funcs.transposeConjugate(mat, np = np)
    return funcs.checkIdentity(iden, eps = eps)
