import CTL.funcs.funcs as funcs
import warnings
def isIsometry(tensor, labels, eps = 1e-10, warningFlag = False):
    """
    Decide if a tensor is Isometry for given labels.

    Parameters
    ----------
    tensor : Tensor

    labels : list of str
        Part of the labels in tensor
    eps : float, default 1e-10
        The upper bound of the error to be ignored when checking the result tensor is identity.
    warningFlag : bool, default False
        If True, then will give a warning message if labels contain all labels of tensor(so cannot give a decision whether it is an isometry, and return True whatever warningFlag is).
    
    Returns
    -------
    bool
        Whether the tensor is an isometry for given labels, up to eps.
    """
    funcName = 'CTL.tensor.tensorFunc.isIsometry'
    assert (tensor.labelsInTensor(labels)), funcs.errorMessage("labels {} is not in {}.".format(labels, tensor), location = funcName)
    if (tensor.tensorLikeFlag):
        if (warningFlag):
            warnings.warn(funcs.warningMessage(warn = 'isIsometry cannot work on tensorLike object {}, return True by default.'.format(tensor), location = funcName))
        return True
    mat = tensor.toMatrix(cols = labels)
    np = tensor.xp
    iden = mat @ funcs.transposeConjugate(mat, np = np)
    return funcs.checkIdentity(iden, eps = eps)
