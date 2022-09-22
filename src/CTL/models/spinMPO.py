import CTL.funcs.xplib as xplib
import CTL.funcs.funcs as funcs
import CTL.funcs.pauli as pauli

from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.link import makeLink
from CTL.examples.MPO import FreeBoundaryMPO

# def HeisenbergMPO(n, J = 1.0, constant = 0):
#     # H = -J(S_xS_x + S_yS_y + S_zS_z)
#     # return an MPO
#     location = 'CTL.models.SpinMPO.HeisenbergMPO'
#     if n < 2:
#         raise ValueError(funcs.errorMessage("n in Heisenberg model cannot be lower than 2, got {}".format(n), location = location))
    
#     leftTensorData = xplib.xp.zeros((5, 2, 2), dtype = xplib.xp.complex128)
#     rightTensorData = xplib.xp.zeros((5, 2, 2), dtype = xplib.xp.complex128)

#     leftTensorData[0] = constant * pauli.identity()
#     leftTensorData[1] = pauli.identity()
#     leftTensorData[2] = pauli.sigmaX()
#     leftTensorData[3] = pauli.sigmaY()
#     leftTensorData[4] = pauli.sigmaZ()

#     rightTensorData[0] = pauli.identity()
#     rightTensorData[2] = -J * pauli.sigmaX()
#     rightTensorData[3] = -J * pauli.sigmaY()
#     rightTensorData[4] = -J * pauli.sigmaZ()

#     leftTensor = Tensor(shape = (5, 2, 2), labels = ['r', 'u', 'd'], data = leftTensorData)
#     rightTensor = Tensor(shape = (5, 2, 2), labels = ['l', 'u', 'd'], data = rightTensorData)

#     tensors = [leftTensor]
#     if n > 2:
#         centerTensorData = xplib.xp.zeros((5, 5, 2, 2), dtype = xplib.xp.complex128)

#         centerTensorData[0, 0] = pauli.identity()
#         centerTensorData[1, 1] = pauli.identity()
#         centerTensorData[1, 2] = pauli.sigmaX()
#         centerTensorData[1, 3] = pauli.sigmaY()
#         centerTensorData[1, 4] = pauli.sigmaZ()

#         centerTensorData[2, 0] = -J * pauli.sigmaX()
#         centerTensorData[3, 0] = -J * pauli.sigmaY()
#         centerTensorData[4, 0] = -J * pauli.sigmaZ()

#         centerTensor = Tensor(shape = (5, 5, 2, 2), labels = ['l', 'r', 'u', 'd'], data = centerTensorData)
#         for i in range(n - 2):
#             tensors.append(centerTensor)

#     tensors.append(rightTensor)
#     for i in range(n - 1):
#         makeLink('r', 'l', tensors[i], tensors[i + 1])
#     return FreeBoundaryMPO(tensorList = tensors, chi = 5)

def nearestNeighborInteractionMPO(n, interaction):
    # interaction: [('z', 'z', 1.0), ('x', 'y', -1.0,)... (None, 'x', 1.0).., (None, None, 2.0)]
    # H = constant + 2.0 + 1.0 * SzSz + (-1.0) * SxSy ... + 1.0 * Sx

    # for two-term case: the first term needs to be contained in vector
    # for one-term case: can be contained in tensor[1, 0]
    '''
    Build Spin 1/2 MPO based on given form of interaction

    Parameters
    ----------
    n : int
        The number of sites in the MPO.
    interaction : str or list of tuples of (str or None, str or None, xplib.xp.complex128)
        The interaction Hamiltonian. Support string input such as "SxSx + SySy + 2SzSz + 3Sz + 5", containing two-body interaction, one-body term(external magnetic field), constant.
        If it is list, then the form should be ('x', 'x', 1.0) for two-body interaction, (None, 'x', 1.0) for one-body term, and (None, None, 3.0) for constant.

    Returns
    -------
    FreeBoundaryMPO
        Open boundary Matrix Product Operator(MPO) of given interaction Hamiltonian.
    '''

    location = 'CTL.models.SpinMPO.nearestNeighborInteractionMPO'
    constant = 0.0
    if n < 2:
        raise ValueError(funcs.errorMessage("n in nearest neighbor interaction MPO cannot be lower than 2, got {}".format(n), location = location))

    if isinstance(interaction, str):
        interaction = interactionTranslation(interaction)

    twoSiteInteraction = dict()
    oneSiteInteraction = dict()

    for term in interaction:
        spinA, spinB, J = term
        if (spinA is None) and (spinB is None):
            constant += J
            continue
        if spinB not in ['x', 'y', 'z']:
            raise ValueError(funcs.errorMessage("Only x/y/z spin operators can be dealt with, {} obtained".format(spinB), location = location))

        if spinA is not None:
            if spinA not in ['x', 'y', 'z']:
                raise ValueError(funcs.errorMessage("Only x/y/z spin operators can be dealt with, {} obtained".format(spinA), location = location))
            if spinA not in twoSiteInteraction:
                twoSiteInteraction[spinA] = []
            twoSiteInteraction[spinA].append((spinB, J))
        else:
            if spinB not in oneSiteInteraction:
                oneSiteInteraction[spinB] = []
            oneSiteInteraction[spinB].append(J)

    sigmaFuncs = {
        'x': pauli.sigmaX,
        'y': pauli.sigmaY, 
        'z': pauli.sigmaZ, 
        'i': pauli.identity
    }

    twoSiteOperatorList = list(twoSiteInteraction.keys())
    chi = len(twoSiteOperatorList) + 2

    leftTensorData = xplib.xp.zeros((chi, 2, 2), dtype = xplib.xp.complex128)
    rightTensorData = xplib.xp.zeros((chi, 2, 2), dtype = xplib.xp.complex128)

    leftTensorData[0] += constant * pauli.identity()
    leftTensorData[1] += pauli.identity()
    
    for i, op in enumerate(twoSiteOperatorList):
        leftTensorData[i + 2] += sigmaFuncs[op]()

    rightTensorData[0] = pauli.identity()
    for op in oneSiteInteraction:
        for J in oneSiteInteraction[op]:
            rightTensorData[1] += sigmaFuncs[op]() * J

    for i, op in enumerate(twoSiteOperatorList):
        for op2, J in twoSiteInteraction[op]:
            rightTensorData[i + 2] += J * sigmaFuncs[op2]()

    singleSiteTensor = xplib.xp.copy(rightTensorData[1])
    nextSiteTensor = xplib.xp.copy(rightTensorData[2:])

    leftTensor = Tensor(shape = (chi, 2, 2), labels = ['r', 'u', 'd'], data = leftTensorData)
    rightTensor = Tensor(shape = (chi, 2, 2), labels = ['l', 'u', 'd'], data = rightTensorData)

    tensors = [leftTensor]
    if n > 2:
        centerTensorData = xplib.xp.zeros((chi, chi, 2, 2), dtype = xplib.xp.complex128)

        centerTensorData[0, 0] = pauli.identity()

        centerTensorData[1, 0] = xplib.xp.copy(singleSiteTensor)
        centerTensorData[1, 1] = pauli.identity()

        for i, op in enumerate(twoSiteOperatorList):
            centerTensorData[1, i + 2] = sigmaFuncs[op]()

        for i in range(chi - 2):
            centerTensorData[i + 2, 0] = xplib.xp.copy(nextSiteTensor[i])

        centerTensor = Tensor(shape = (chi, chi, 2, 2), labels = ['l', 'r', 'u', 'd'], data = centerTensorData)
        for i in range(n - 2):
            tensors.append(centerTensor)

    tensors.append(rightTensor)
    for i in range(n - 1):
        makeLink('r', 'l', tensors[i], tensors[i + 1])
    return FreeBoundaryMPO(tensorList = tensors, chi = chi)

def HeisenbergMPO(n, J = 1.0, constant = 0):
    '''
    Create an MPO representing Hamiltonian H = -J(S_x S_x + S_y S_y + S_z S_z) + constant
    Parameters
    ----------
    n : int
        Number of sites.
    J : xplib.xp.complex128, default 1 + 0j
        Interaction factor.
    constant : xplib.xp.complex128, default 0 + 0j
        Constant in Hamiltonian.

    Returns
    -------
    FreeBoundaryMPO
        Open boundary Matrix Product Operator(MPO) representing given Hamiltonian.
    '''
    return nearestNeighborInteractionMPO(
        n = n, 
        interaction = [
            ('x', 'x', -J), 
            ('y', 'y', -J), 
            ('z', 'z', -J), 
            (None, None, constant)
        ]
    )

def IsingMPO(n, J = 1.0, constant = 0):
    '''
    Create an MPO representing Hamiltonian H = -JS_zS_z + constant
    Parameters
    ----------
    n : int
        Number of sites.
    J : xplib.xp.complex128, default 1 + 0j
        Interaction factor.
    constant : xplib.xp.complex128, default 0 + 0j
        Constant in Hamiltonian.

    Returns
    -------
    FreeBoundaryMPO
        Open boundary Matrix Product Operator(MPO) representing given Hamiltonian.
    '''
    return nearestNeighborInteractionMPO(
        n = n, 
        interaction = [
            ('z', 'z', -J),
            (None, None, constant)
        ]
    )

def termTranslation(term, sign = '+'):
    location = 'CTL.models.spinMPO.termTranslation'
    if sign is None:
        sign = '+'
    if not (sign in ['+', '-']):
        raise ValueError(funcs.errorMessage('sign can only be +/-, {} got'.format(sign), location = location))

    if sign == '+':
        sign = 1.0
    else:
        sign = -1.0
    term = term.strip()
    
    sIndex = term.find('S')
    if sIndex == -1:
        try:
            compactTerm = term.replace(' ', '')
            J = xplib.xp.complex128(compactTerm)
        except:
            raise ValueError(funcs.errorMessage('term {} does not contain S, but also not a constant'.format(term), location = location))
        return (None, None, J)
    
    factor = term[:sIndex].strip()
    while (len(factor) > 0):
        if (factor[0] == '(' and factor[-1] == ')'):
            factor = factor[1:-1].strip()
        elif factor[-1] == '*':
            factor = factor[:-1].strip()
        else:
            break
    
    if len(factor) == 0:
        J = 1.0 + 0.0j
    else:
        try:
            compactTerm = factor.replace(' ', '')
            J = xplib.xp.complex128(compactTerm)
        except:
            raise ValueError(funcs.errorMessage('factor {} is not a constant'.format(factor), location = location))

    spinTerms = term[sIndex:].split('S')[1:]
    spinTerms = [x.strip() for x in spinTerms]
    if len(spinTerms) > 2:
        raise ValueError(funcs.errorMessage('term {} contains more than 2 spins'.format(term), location = location))

    if len(spinTerms) == 0:
        raise ValueError(funcs.errorMessage('term {} is not valid'.format(term), location = location))

    if len(spinTerms) == 1:
        spin1, spin2 = None, spinTerms[0]
    else:
        spin1, spin2 = spinTerms
    
    if (spin1 is not None) and (spin1 not in ['x', 'y', 'z']):
        raise ValueError(funcs.errorMessage('spin {} is not a valid direction'.format(spin1), location = location))
    if (spin2 not in ['x', 'y', 'z']):
        raise ValueError(funcs.errorMessage('spin {} is not a valid direction'.format(spin2), location = location))

    return (spin1, spin2, sign * J)

def interactionTranslation(intStr = '-1.0SzSz'):
    intStr = intStr.replace('i', 'j')
    location = 'CTL.models.spinMPO.interactionTranslation'
    currentSign = None
    currentLeftPar = 0

    currS = ''
    res = []
    for c in intStr:
        if c in ['+', '-'] and currentLeftPar == 0:
            currS = currS.strip()
            if len(currS) > 0:
                res.append(termTranslation(currS, sign = currentSign))
            currentSign = c
            currS = ''
        else:
            currS += c
            if c == '(':
                currentLeftPar += 1
            elif c == ')':
                currentLeftPar -= 1

    if currentLeftPar != 0:
        raise ValueError(funcs.errorMessage('Parentheses are not compatible in {}'.format(intStr), location = location))
    
    if len(currS) > 0:
        res.append(termTranslation(currS, sign = currentSign))

    return res