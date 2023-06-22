import CTL.funcs.funcs as funcs
from CTL.funcs.decompose import SVDDecomposition
from CTL.tensor.tensor import Tensor 
from CTL.tensor.diagonalTensor import DiagonalTensor
from CTL.tensor.leg import Leg
from CTL.tensor.contract.link import makeLink

# import numpy as np
import CTL.funcs.xplib as xplib
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
    # np = tensor.xp
    iden = mat @ funcs.transposeConjugate(mat)
    return funcs.checkIdentity(iden, eps = eps)

def tensorSVDDecomposition(a, rows = None, cols = None, innerLabels = None, preserveLegs = False, chi = 16, errorOrder = 2, keepdim = False, keepLink = False):
    """
    Decompose a tensor into two isometry tensors and one diagonal tensor, according to SVD decomposition. Namely, a = u @ s @ v.

    Parameters
    ----------
    a : Tensor

    rows, cols : None or list of str
        The labels of tensor a that form the rows and cols of the matrix to be decomposed. For more information, please check Tensor.toMatrix.
    innerLabels : None or length-2 tuple of str
        If not None, then gives the two labels for tensor u and v pointing to s, otherwise they are set to be "inner:" + auto-generated name from combined rows and cols. Note that, the legs of tensor s are always named as auto-generated labels, so that for rank-2 tensor a, the "form" will not change between a and s.
    preserveLegs(not activated yet) : bool, default False, currently always False
        If True, then the outer legs for u and v will be kept from the legs of a. This is somehow dangerous, since one leg is used twice, both in u(v) and a, so use this only when len(rows) and len(cols) is 1, and when you want to drop out tensor a.
        The usage of this parameter will lead to important changes in Tensor deduction logic. Since the usage is not found yet, we temporarily set this flag to be always False.
    chi : int, default 16
        Maximum bond dimension of inner legs.
    errorOrder : int, default 2
        The order of error in singular value decomposition. The error will be calculated as (s[chi:] ** errorOrder).sum() / (s ** errorOrder).sum().
    keepdim: bool, default False
        Whether to keep the outer legs of a in original shape(may contain several legs). If True, then the legs will have the same name / shape as the original tensor, and if preserveLegs is also True, then the leg objects will be just used in new tensor.
    keepLink: bool, default False
        Whether to connect the output tensors, i.e. u, s, vh. If True, then they will be connected, and the tensor contraction will give just the original tensor(with approximation).

    Returns
    -------
    d : dict of keys {"u", "s", "v", "error"}
        d["u"], d["v"] : rank-2 Tensor(keepdim = False), or rank-(N + 1) and rank-(M + 1) tensor
        
        d["s"] : rank-2 DiagonalTensor
            a ~= d["u"] @ d["s"] @ d["v"]
        d["error"] : float
            The error from SVD process.
    """

    if not keepdim:
        preserveLegs = False

    funcName = "CTL.tensor.tensorFuncs.tensorSVDDecomposition"

    aMat = a.toMatrix(rows = rows, cols = cols)
    # print('aMat = {}'.format(aMat))
    rowName = '|'.join(rows)
    colName = '|'.join(cols)
    if keepdim:
        rowLegs, colLegs = a.deductRowColumn(rows = rows, cols = cols)
        rowLabels = tuple([leg.name for leg in rowLegs])
        colLabels = tuple([leg.name for leg in colLegs])
        rowShape = tuple([leg.dim for leg in rowLegs])
        colShape = tuple([leg.dim for leg in colLegs])
        if not preserveLegs:
            newRowLegs = [Leg(tensor = None, dim = dim, name = name) for dim, name in zip(rowShape, rowLabels)]
            newColLegs = [Leg(tensor = None, dim = dim, name = name) for dim, name in zip(colShape, colLabels)]
        else:
            newRowLegs = rowLegs
            newColLegs = colLegs

    # rowLeg = None
    # colLeg = None
    # if preserveLegs:
    #     if len(rows) == 1:
    #         rowLeg = a.getLeg(rows[0])
    #     else:
    #         warnings.warn(funcs.warningMessage('cannot preserve leg object when rows is more than one legs {}'.format(rows), location = funcName), RuntimeWarning)

    #     if len(cols) == 1:
    #         colLeg = a.getLeg(cols[0])
    #     else:
    #         warnings.warn(funcs.warningMessage('cannot preserve leg object when columns is more than one legs {}'.format(cols), location = funcName), RuntimeWarning)
    #     # take the leg of row for new tensor

    if (innerLabels is not None):
        assert (isinstance(innerLabels, tuple) and len(innerLabels) == 2), funcs.errorMessage("inner labels can either be None or length-2 tuple, {} obtained.".format(innerLabels), location = funcName)
        rowLabel, colLabel = innerLabels
    else:
        rowLabel = 'inner:' + rowName 
        colLabel = 'inner:' + colName

    u, s, vh, error = SVDDecomposition(aMat, chi = chi, returnSV = True, errorOrder = errorOrder)

    uInnerDim = u.shape[1]
    vInnerDim = vh.shape[0]

    uInnerLeg = Leg(tensor = None, dim = uInnerDim, name = rowLabel)
    vInnerLeg = Leg(tensor = None, dim = vInnerDim, name = colLabel)

    if keepdim:
        uTensor = Tensor(data = u, legs = newRowLegs + [uInnerLeg])
        sTensor = DiagonalTensor(data = s, labels = [rowName, colName])
        vTensor = Tensor(data = vh, legs = [vInnerLeg] + newColLegs)

        if (keepLink):
            makeLink(legA = uInnerLeg, legB = sTensor.getLeg(rowName))
            makeLink(legA = vInnerLeg, legB = sTensor.getLeg(colName))
            
    else:
        uTensor = Tensor(data = u, labels = [rowName, rowLabel])
        sTensor = DiagonalTensor(data = s, labels = [rowName, colName])
        # sTensor = Tensor(data = xplib.xp.diag(s), labels = [rowName, colName])
        vTensor = Tensor(data = vh, labels = [colLabel, colName])
        if (keepLink):
            makeLink(rowLabel, rowName, uTensor, sTensor)
            makeLink(colLabel, colName, vTensor, sTensor)

    return {'u': uTensor, 's': sTensor, 'v': vTensor, 'error': error}

    