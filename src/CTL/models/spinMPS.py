import CTL.funcs.xplib as xplib
import CTL.funcs.funcs as funcs
import CTL.funcs.pauli as pauli

from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.link import makeLink
from CTL.examples.MPS import FreeBoundaryMPS

def createFerroMPS(n):
    # spin 1/2
    location = 'CTL.models.spinMPS.createFerroMPS'
    if n < 2:
        raise ValueError(funcs.errorMessage("n in ferromagnetic MPS cannot be lower than 2, got {}".format(n), location = location))

    leftTensorData = xplib.xp.zeros((1, 2), dtype = xplib.xp.complex128)
    rightTensorData = xplib.xp.zeros((1, 2), dtype = xplib.xp.complex128)

    leftTensorData[0, 0] = 1.0
    rightTensorData[0, 0] = 1.0

    leftTensor = Tensor(shape = (1, 2), labels = ['r', 'o'], data = leftTensorData)
    rightTensor = Tensor(shape = (1, 2), labels = ['l', 'o'], data = rightTensorData)

    tensors = [leftTensor]
    if n > 2:
        centerTensorData = xplib.xp.zeros((1, 1, 2), dtype = xplib.xp.complex128)
        centerTensorData[0, 0, 0] = 1

        centerTensor = Tensor(shape = (1, 1, 2), labels = ['l', 'r', 'o'], data = centerTensorData)
        for i in range(n - 2):
            tensors.append(centerTensor.copy())

    tensors.append(rightTensor)
    for i in range(n - 1):
        makeLink('r', 'l', tensors[i], tensors[i + 1])
    res = FreeBoundaryMPS(tensorList = tensors, chi = 1)
    res.normalize(idx = 0)
    return res
    
def createAntiFerroMPS(n):
    def getNonZeroIndex(i):
        if (i % 2 == 0):
            return 0
        else:
            return 1
    # spin 1/2
    location = 'CTL.models.spinMPS.createAntiFerroMPS'
    if n < 2:
        raise ValueError(funcs.errorMessage("n in ferromagnetic MPS cannot be lower than 2, got {}".format(n), location = location))

    leftTensorData = xplib.xp.zeros((1, 2), dtype = xplib.xp.complex128)
    rightTensorData = xplib.xp.zeros((1, 2), dtype = xplib.xp.complex128)

    leftTensorData[0, 0] = 1.0
    rightTensorData[0, getNonZeroIndex(n - 1)] = 1.0

    leftTensor = Tensor(shape = (1, 2), labels = ['r', 'o'], data = leftTensorData)
    rightTensor = Tensor(shape = (1, 2), labels = ['l', 'o'], data = rightTensorData)

    tensors = [leftTensor]
    if n > 2:
        for i in range(n - 2):
            centerTensorData = xplib.xp.zeros((1, 1, 2), dtype = xplib.xp.complex128)
            centerTensorData[0, 0, getNonZeroIndex(i + 1)] = 1

            centerTensor = Tensor(shape = (1, 1, 2), labels = ['l', 'r', 'o'], data = centerTensorData)
            tensors.append(centerTensor)

    tensors.append(rightTensor)
    for i in range(n - 1):
        makeLink('r', 'l', tensors[i], tensors[i + 1])
    res = FreeBoundaryMPS(tensorList = tensors, chi = 1)
    res.normalize(idx = 0)
    return res