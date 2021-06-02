from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensornetwork.tensordict import TensorDict
from CTL.tensor.tensor import Tensor

from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.contract import contractTensors

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

def squareHorizontalContractFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['ul', 'ur', 'dr', 'dl'])
    
    FTN.addLink('ul', 'd', 'dl', 'u')
    FTN.addLink('ul', 'r', 'ur', 'l')
    FTN.addLink('ur', 'd', 'dr', 'u')
    FTN.addLink('dl', 'r', 'dr', 'l')

    FTN.addLink('ul', 'u', 'ur', 'u')
    FTN.addLink('dl', 'd', 'dr', 'd')

    FTN.addPostOutProduct(['ul-l', 'dl-l'], 'l')
    FTN.addPostOutProduct(['ur-r', 'dr-r'], 'r')
    return FTN

def squareVerticalContractFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['ul', 'ur', 'dr', 'dl'])
    
    FTN.addLink('ul', 'd', 'dl', 'u')
    FTN.addLink('ul', 'r', 'ur', 'l')
    FTN.addLink('ur', 'd', 'dr', 'u')
    FTN.addLink('dl', 'r', 'dr', 'l')

    FTN.addLink('ul', 'l', 'dl', 'l')
    FTN.addLink('ur', 'r', 'dr', 'r')

    FTN.addPostOutProduct(['ul-u', 'ur-u'], 'u')
    FTN.addPostOutProduct(['dl-d', 'dr-d'], 'd')
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



