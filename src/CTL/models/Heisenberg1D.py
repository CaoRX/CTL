import CTL.funcs.xplib as xplib
import CTL.funcs.funcs as funcs
import CTL.funcs.pauli as pauli

from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.link import makeLink
from CTL.examples.MPO import FreeBoundaryMPO

def HeisenbergMPO(n, J = 1.0, constant = 0):
    # H = -J(S_xS_x + S_yS_y + S_zS_z)
    # return an MPO
    location = 'CTL.models.Heisenberg1D.HeisenbergMPO'
    if n < 2:
        raise ValueError(funcs.errorMessage("n in Heisenberg model cannot be lower than 2, got {}".format(n), location = location))
    
    leftTensorData = xplib.xp.zeros((5, 2, 2), dtype = xplib.xp.complex128)
    rightTensorData = xplib.xp.zeros((5, 2, 2), dtype = xplib.xp.complex128)

    leftTensorData[0] = constant * pauli.identity()
    leftTensorData[1] = pauli.identity()
    leftTensorData[2] = pauli.sigmaX()
    leftTensorData[3] = pauli.sigmaY()
    leftTensorData[4] = pauli.sigmaZ()

    rightTensorData[0] = pauli.identity()
    rightTensorData[2] = -J * pauli.sigmaX()
    rightTensorData[3] = -J * pauli.sigmaY()
    rightTensorData[4] = -J * pauli.sigmaZ()

    leftTensor = Tensor(shape = (5, 2, 2), labels = ['r', 'u', 'd'], data = leftTensorData)
    rightTensor = Tensor(shape = (5, 2, 2), labels = ['l', 'u', 'd'], data = rightTensorData)

    tensors = [leftTensor]
    if n > 2:
        centerTensorData = xplib.xp.zeros((5, 5, 2, 2), dtype = xplib.xp.complex128)

        centerTensorData[0, 0] = pauli.identity()
        centerTensorData[1, 1] = pauli.identity()
        centerTensorData[1, 2] = pauli.sigmaX()
        centerTensorData[1, 3] = pauli.sigmaY()
        centerTensorData[1, 4] = pauli.sigmaZ()

        centerTensorData[2, 0] = -J * pauli.sigmaX()
        centerTensorData[3, 0] = -J * pauli.sigmaY()
        centerTensorData[4, 0] = -J * pauli.sigmaZ()

        centerTensor = Tensor(shape = (5, 5, 2, 2), labels = ['l', 'r', 'u', 'd'], data = centerTensorData)
        for i in range(n - 2):
            tensors.append(centerTensor)

    tensors.append(rightTensor)
    for i in range(n - 1):
        makeLink('r', 'l', tensors[i], tensors[i + 1])
    return FreeBoundaryMPO(tensorList = tensors, chi = 5)