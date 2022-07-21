# use MPO to optimize the ground state of Heisenberg model in MPS
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'src'))

import CTL

from CTL.examples.MPS import FreeBoundaryMPS, createRandomMPS
from CTL.examples.MPO import FreeBoundaryMPO, identityMPO
import CTL.funcs.funcs as funcs
import numpy as np
from CTL.tensor.contract.link import makeLink
from CTL.tensor.tensor import Tensor
import CTL.funcs.pauli as pauli
import CTL.funcs.xplib as xplib
from CTL.models.Heisenberg1D import HeisenbergMPO
from CTL.models.spinMPS import createAntiFerroMPS, createFerroMPS
from CTL.models.Ising import IsingMPO

if __name__ == '__main__':
    # initialMPS = HeisenbergRandomInitialMPS(n = 5, chi = 10)
    CTL.turnOffOutProductWarning()
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

    J = 1.0
    Heisenberg = HeisenbergMPO(n = 5, J = J)
    ip2 = Heisenberg.innerProduct(mpsU = initialMPS, mpsD = initialMPS)
    print('<psi | H_Heisenberg | psi> = {}'.format(ip2.a))

    ferroMPS = createFerroMPS(n = 5)
    ipFerro = Heisenberg.innerProduct(mpsU = ferroMPS, mpsD = ferroMPS)
    print('<psi_ferro | H_Heisenberg | psi_ferro> = {} for J = {}'.format(ipFerro.a, J))

    antiFerroMPS = createAntiFerroMPS(n = 5)
    ipAntiFerro = Heisenberg.innerProduct(mpsU = antiFerroMPS, mpsD = antiFerroMPS)
    print('<psi_antiferro | H_Heisenberg | psi_antiferro> = {} for J = {}'.format(ipAntiFerro.a, J))

    J = -1.0
    antiFerroHeisenberg = HeisenbergMPO(n = 5, J = J)
    afipFerro = antiFerroHeisenberg.innerProduct(mpsU = ferroMPS, mpsD = ferroMPS)
    print('<psi_ferro | H_af_Heisenberg | psi_ferro> = {} for J = {}'.format(afipFerro.a, J))

    constant = -4.0
    constantMPO = HeisenbergMPO(n = 5, J = 1.0, constant = constant)
    ipFerro = constantMPO.innerProduct(mpsU = ferroMPS, mpsD = ferroMPS)
    print('<psi_ferro | {} + H_Heisenberg | psi_ferro> = {} for J = {}'.format(constant, ipFerro.a, J))

    ipAntiFerro = constantMPO.innerProduct(mpsU = antiFerroMPS, mpsD = antiFerroMPS)
    print('<psi_antiferro | {} + H_Heisenberg | psi_antiferro> = {} for J = {}'.format(constant, ipAntiFerro.a, J))
    
    initialMPS = createRandomMPS(n = 5, chi = 10, dim = 2)
    
    initialH = Heisenberg.innerProduct(mpsU = initialMPS, mpsD = initialMPS).a

    currMPS = initialMPS
    step = 20
    print('initial H = {}'.format(initialH))
    for i in range(step):

        newMPS = constantMPO.applyToMPS(mps = currMPS)
        newMPS.normalize(idx = 0)

        newH = Heisenberg.innerProduct(mpsU = newMPS, mpsD = newMPS).a
        print('new H after apply (H + {}) = {}'.format(constant, newH))

        currMPS = newMPS

    Ising = IsingMPO(n = 5, J = 1.0)
    print('<psi_init | H_Ising | psi_init> = {}'.format(Ising.innerProduct(mpsU = initialMPS, mpsD = initialMPS).a))