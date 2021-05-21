from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensor.tensor import Tensor

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

