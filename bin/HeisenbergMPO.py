# use MPO to optimize the ground state of Heisenberg model in MPS
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'src'))

from CTL.examples.MPS import FreeBoundaryMPS, createRandomMPS
from CTL.examples.MPO import FreeBoundaryMPO, identityMPO
import CTL.funcs.funcs as funcs
import numpy as np
from CTL.tensor.contract.link import makeLink
from CTL.tensor.tensor import Tensor
        
def IsingMPO(n, J = 1.0):
    pass
def HeisenbergMPO(n, J = 1.0):
    # H = -J(S_xS_x + S_yS_y + S_zS_z)
    # return an MPO
    pass

if __name__ == '__main__':
    # initialMPS = HeisenbergRandomInitialMPS(n = 5, chi = 10)
    initialMPS = createRandomMPS(n = 5, chi = 10, dim = 2)

    print(initialMPS.norm())
    print(initialMPS.checkMPSProperty())

    identity = identityMPO(dim = 2, n = 5)
    newMPS = identity.applyToMPS(initialMPS)
    # print(newMPS)
    print("new norm = {}".format(newMPS.norm()))

    newMPS = identity.applyToMPS(initialMPS)
    # print(newMPS)
    print("new norm = {}".format(newMPS.norm()))

    ip = identity.innerProduct(mpsU = initialMPS, mpsD = initialMPS)
    print('inner product = {}'.format(ip.a))


