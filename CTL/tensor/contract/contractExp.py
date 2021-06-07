from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensornetwork.tensordict import TensorDict
from CTL.tensor.tensor import Tensor

from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.contract import contractTensors
import CTL.funcs.funcs as funcs

def makeTriangleTensorDict(a):
    return TensorDict({'u': a, 'l': a, 'r': a})

def makeSquareTensorDict(a, b = None):
    if (b is None):
        b = a 
    return TensorDict({'ul': a, 'ur': b, 'dr': a, 'dl': b})

def triangleContractFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['u', 'l', 'r'])
    FTN.addLink('u', '2', 'l', '3')
    FTN.addLink('u', '3', 'r', '2')
    FTN.addLink('l', '2', 'r', '3')
    
    FTN.addPostNameChange('u', '1', '1')
    FTN.addPostNameChange('l', '1', '2')
    FTN.addPostNameChange('r', '1', '3')

    return FTN

def squareContractFTN():
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

def triangleTensorTrace(a, b):
    tensorA = a.copy()
    tensorB = b.copy()
    
    for label in ['1', '2', '3']:
        makeLink(label, label, tensorA = tensorA, tensorB = tensorB)
    
    res = contractTensors(tensorA, tensorB)
    return res

def squareHorizontalContractFTN(d):
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


