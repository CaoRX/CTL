from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensornetwork.tensordict import TensorDict
from CTL.tensor.tensor import Tensor

from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.contract import contractTwoTensors
import CTL.funcs.funcs as funcs

def makeTriangleTensorDict(a):
    """
    Make a tensor dict of a triangle, from a single tensor.
    
    Parameters
    ----------
    a : Tensor
        The tensor to be put on the three corners of the triangle.

    Returns
    -------
    dict
        A dict with 3 keys, representing a triangle.
    """
    return TensorDict({'u': a, 'l': a, 'r': a})

def makeSquareTensorDict(a, b = None):
    """
    Make a tensor dict of a square, from one or two tensor.
    
    Parameters
    ----------
    a : Tensor

    b : Tensor, optional
        If given, then the square will be [ab, ba], otherwise [aa, aa]
    
    Returns
    -------
    dict
        A dict with 4 keys, representing a square.
    """
    if (b is None):
        b = a 
    return TensorDict({'ul': a, 'ur': b, 'dr': a, 'dl': b})

def triangleContractFTN():
    """
    Make a finite tensor network for triangle tensor contraction.

    Returns
    -------
    FTN : FiniteTensorNetwork
        A FiniteTensorNetwork that, takes three tensors(in form of makeTriangleTensorDict), legs of which are named as ["1", "2", "3"], contract them.
    
    """
    FTN = FiniteTensorNetwork(tensorNames = ['u', 'l', 'r'])
    FTN.addLink('u', '2', 'l', '3')
    FTN.addLink('u', '3', 'r', '2')
    FTN.addLink('l', '2', 'r', '3')
    
    FTN.addPostNameChange('u', '1', '1')
    FTN.addPostNameChange('l', '1', '2')
    FTN.addPostNameChange('r', '1', '3')

    return FTN

def squareContractFTN():
    """
    Make a finite tensor network for square tensor contraction.

    Returns
    -------
    FTN : FiniteTensorNetwork
        A FiniteTensorNetwork that, takes 4 tensors(in form of makeSquareTensorDict), legs of which are named as ['u', 'd', 'l', 'r'], contract them.
    
    """
    FTN = FiniteTensorNetwork(tensorNames = ['ul', 'ur', 'dr', 'dl'])
    FTN.addLink('ul', 'd', 'dl', 'u')
    FTN.addLink('ul', 'r', 'ur', 'l')
    FTN.addLink('ur', 'd', 'dr', 'u')
    FTN.addLink('dl', 'r', 'dr', 'l')

    FTN.addPostOutProduct(['ul-u', 'ur-u'], 'u')
    FTN.addPostOutProduct(['ul-l', 'dl-l'], 'l')
    FTN.addPostOutProduct(['dl-d', 'dr-d'], 'd')
    FTN.addPostOutProduct(['ur-r', 'dr-r'], 'r')
    
    return FTN

def squareContractOutFTN():
    """
    Make a finite tensor network for square tensor contraction, where tensors are having inner legs and outer legs named "o".

    Returns
    -------
    FTN : FiniteTensorNetwork
        A FiniteTensorNetwork that, takes 4 tensors(in form of makeSquareTensorDict), legs of which are named as ['u'/'d', 'l'/'r', 'o'], contract them.
    
    """
    FTN = FiniteTensorNetwork(tensorNames = ['ul', 'ur', 'dr', 'dl'])
    
    FTN.addLink('ul', 'd', 'dl', 'u')
    FTN.addLink('ul', 'r', 'ur', 'l')
    FTN.addLink('ur', 'd', 'dr', 'u')
    FTN.addLink('dl', 'r', 'dr', 'l')

    FTN.addPostNameChange('ul', 'o', 'u')
    FTN.addPostNameChange('ur', 'o', 'r')
    FTN.addPostNameChange('dl', 'o', 'l')
    FTN.addPostNameChange('dr', 'o', 'd')

    return FTN

def CTMRGHEdgeExtendFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['p', 'w'])

    FTN.addLink('p', 'u', 'w', 'd')

    FTN.addPostOutProduct(['p-l', 'w-l'], 'l')
    FTN.addPostOutProduct(['p-r', 'w-r'], 'r')
    # FTN.addPostNameChange('p', 'l', 'ld')
    # FTN.addPostNameChange('p', 'r', 'rd')
    # FTN.addPostNameChange('w', 'l', 'lu')
    # FTN.addPostNameChange('w', 'r', 'ru')

    return FTN

def CTMRGVEdgeExtendFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['p', 'w'])

    FTN.addLink('p', 'r', 'w', 'l')
    
    FTN.addPostOutProduct(['p-u', 'w-u'], 'u')
    FTN.addPostOutProduct(['p-d', 'w-d'], 'd')
    # FTN.addPostNameChange('p', 'u', 'ul')
    # FTN.addPostNameChange('p', 'd', 'dl')
    # FTN.addPostNameChange('w', 'u', 'ur')
    # FTN.addPostNameChange('w', 'd', 'dr')

    return FTN

def CTMRGCornerExtendFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['c', 'ph', 'pv', 'w'])

    FTN.addLink('c', 'u', 'pv', 'd')
    FTN.addLink('c', 'r', 'ph', 'l')
    FTN.addLink('pv', 'r', 'w', 'l')
    FTN.addLink('ph', 'u', 'w', 'd')

    FTN.addPostOutProduct(['pv-u', 'w-u'], 'u')
    FTN.addPostOutProduct(['ph-r', 'w-r'], 'r')

    return FTN

def CTMRGHEdgeBuildFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['p', 'al', 'ar'])
    # our a[l, r] is only for left part; for right part of a, we need to still point "r" to p
    FTN.addLink('al', 'r', 'p', 'l')
    FTN.addLink('ar', 'r', 'p', 'r')

    FTN.addPostNameChange('ar', 'l', 'r')
    return FTN

def CTMRGVEdgeBuildFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['p', 'au', 'ad'])
    # our a[u, d] is only for down part; for up part of a, we need to still point "u" to p
    FTN.addLink('ad', 'u', 'p', 'd')
    FTN.addLink('au', 'u', 'p', 'u')

    FTN.addPostNameChange('au', 'd', 'u')
    return FTN

def CTMRGEvenZFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['alu', 'ald', 'aru', 'ard'])
    FTN.addLink('alu', 'u', 'ald', 'u')
    FTN.addLink('aru', 'u', 'ard', 'u')
    FTN.addLink('alu', 'r', 'aru', 'r')
    FTN.addLink('ald', 'r', 'ard', 'r')

    return FTN

def CTMRGOddFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['alu', 'ald', 'aru', 'ard', 'pu', 'pd', 'pl', 'pr', 'w'])

    FTN.addLink('pu', 'l', 'alu', 'r')
    FTN.addLink('pu', 'r', 'aru', 'r')
    FTN.addLink('pd', 'l', 'ald', 'r')
    FTN.addLink('pd', 'r', 'ard', 'r')

    FTN.addLink('pl', 'u', 'alu', 'u')
    FTN.addLink('pl', 'd', 'ald', 'u')
    FTN.addLink('pr', 'u', 'aru', 'u')
    FTN.addLink('pr', 'd', 'ard', 'u')

    FTN.addLink('pl', 'r', 'w', 'l')
    FTN.addLink('pr', 'r', 'w', 'r')
    FTN.addLink('pd', 'u', 'w', 'd')
    FTN.addLink('pu', 'u', 'w', 'u')
    
    return FTN

def triangleTensorTrace(a, b):
    """
    Take the trace of two triangle tensors, usually work for the end of the RG of triangular lattice.

    Parameters
    ----------
    a, b : Tensor
        Two tensors of the triangular shape(namely, 3 legs ['1', '2', '3']).
    
    Returns
    -------
    Tensor
        The trace tensor of the two tensors. If a and b only have legs ['1', '2', '3'], then shapeless(so the value can be obtained as res.single()), otherwise the tensor of the remaining legs.
    """
    tensorA = a.copy()
    tensorB = b.copy()

    bonds = []
    
    for label in ['1', '2', '3']:
        bonds += makeLink(label, label, tensorA = tensorA, tensorB = tensorB)
    
    res = contractTwoTensors(tensorA, tensorB, bonds = bonds)
    return res

def squareHorizontalContractFTN(d):
    # TODO: add docstrings for following functions
    funcs.assertInSet(d, ['l', 'r'], 'horizontal direction')
    opd = funcs.oppositeSingleDirection(d)
    FTN = FiniteTensorNetwork(tensorNames = ['ul', 'ur', 'dr', 'dl'])
    
    FTN.addLink('ul', 'd', 'dl', 'u')
    FTN.addLink('ul', opd, 'ur', opd)
    FTN.addLink('ur', 'd', 'dr', 'u')
    FTN.addLink('dl', opd, 'dr', opd)

    FTN.addLink('ul', 'u', 'ur', 'u')
    FTN.addLink('dl', 'd', 'dr', 'd')

    FTN.addPostOutProduct(['u' + d + '-' + d, 'd' + d + '-' + d], d)
    FTN.addPostOutProduct(['u' + opd + '-' + d, 'd' + opd + '-' + d], opd)
    return FTN

def squareVerticalContractFTN(d):
    funcs.assertInSet(d, ['u', 'd'], 'vertical direction')
    opd = funcs.oppositeSingleDirection(d)
    FTN = FiniteTensorNetwork(tensorNames = ['ul', 'ur', 'dr', 'dl'])
    
    FTN.addLink('ul', opd, 'dl', opd)
    FTN.addLink('ul', 'r', 'ur', 'l')
    FTN.addLink('ur', opd, 'dr', opd)
    FTN.addLink('dl', 'r', 'dr', 'l')

    FTN.addLink('ul', 'l', 'dl', 'l')
    FTN.addLink('ur', 'r', 'dr', 'r')

    FTN.addPostOutProduct([d + 'l-' + d, d + 'r-' + d], d)
    FTN.addPostOutProduct([opd + 'l-' + d, opd + 'r-' + d], opd)
    return FTN

def HOTRGHorizontalContractFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['u', 'd', 'l', 'r'])
    FTN.addLink('u', 'l', 'l', 'u')
    FTN.addLink('u', 'r', 'r', 'u')
    FTN.addLink('d', 'l', 'l', 'd')
    FTN.addLink('d', 'r', 'r', 'd')

    FTN.addLink('u', 'd', 'd', 'u')

    FTN.addPostNameChange('l', 'o', 'l')
    FTN.addPostNameChange('r', 'o', 'r')

    return FTN 

def HOTRGVerticalContractFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['u', 'd', 'l', 'r'])
    FTN.addLink('u', 'l', 'l', 'u')
    FTN.addLink('u', 'r', 'r', 'u')
    FTN.addLink('d', 'l', 'l', 'd')
    FTN.addLink('d', 'r', 'r', 'd')

    FTN.addLink('l', 'r', 'r', 'l')

    FTN.addPostNameChange('u', 'o', 'u')
    FTN.addPostNameChange('d', 'o', 'd')

    return FTN 

def squareTrace(a):
    return a.trace(rows = ['u', 'l'], cols = ['d', 'r'])

def EvenblyTNRQEnvFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['uul', 'uur', 'udl', 'udr', 'dul', 'dur', 'ddl', 'ddr'])

    FTN.addLink('udl', 'r', 'dul', 'r')
    FTN.addLink('udl', 'd', 'dul', 'd')
    FTN.addLink('udr', 'r', 'dur', 'r')
    FTN.addLink('udr', 'd', 'dur', 'd')

    FTN.addLink('udl', 'l', 'udr', 'l')
    FTN.addLink('dul', 'l', 'dur', 'l')
    FTN.addLink('uur', 'l', 'ddr', 'l')
    FTN.addLink('uur', 'u', 'ddr', 'u')

    FTN.addLink('udr', 'u', 'uur', 'd')
    FTN.addLink('dur', 'u', 'ddr', 'd')
    FTN.addLink('uul', 'r', 'uur', 'r')
    FTN.addLink('uul', 'd', 'udl', 'u')
    FTN.addLink('ddl', 'r', 'ddr', 'r')
    FTN.addLink('ddl', 'd', 'dul', 'u')

    FTN.addPostOutProduct(['uul-l', 'uul-u'], '1')
    FTN.addPostOutProduct(['ddl-l', 'ddl-u'], '2')

    return FTN

def selfTrace(tensor):
    a, b = tensor.copyN(2)
    for leg1, leg2 in zip(a.legs, b.legs):
        makeLink(leg1, leg2)

    return a @ b
    
