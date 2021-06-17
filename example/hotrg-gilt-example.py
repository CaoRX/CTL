import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# import the functions and Classes we will use
# the import now is a little troublesome to find where the function you want is
# it is planned to be improved, maybe like numpy structure(so all functions can be used with CTL.xx?)
from CTL.tensor.tensor import Tensor 
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
from CTL.tensornetwork.tensordict import TensorDict
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.optimalContract import contractTensorList, generateOptimalSequence, contractWithSequence

from CTL.models.Ising import squareIsingTensor, infiniteIsingExactM
from CTL.examples.impurity import ImpurityTensorNetwork
from CTL.examples.HOTRG import HOTRG

import numpy as np

def HOTRGGiltTN():
    ATensorNames = ["aul", "aur", "adl", "adr"]
    QTensorNames = ['qbll', 'qbrl', 'qall', 'qarl', 'qblr', 'qbrr', 'qalr', 'qarr', 'qbu', 'qbd', 'qau', 'qad'] # i(circle), o(line)
    wTensorNames = ['wll', 'wlr', 'wrl', 'wrr'] # u, d, o
    vTensorNames = ['vu', 'vd'] # l, r, o

    tensorNames = ATensorNames + QTensorNames + wTensorNames + vTensorNames

    FTN = FiniteTensorNetwork(tensorNames = tensorNames)

    # FTN.addLink('aul', 'adl', 'd', 'u')
    # FTN.addLink('aur', 'adr', 'd', 'u')

    # up / down bonds
    FTN.addLink('vu', 'l', 'qbu', 'o')
    FTN.addLink('vu', 'r', 'qau', 'o')
    FTN.addLink('vd', 'l', 'qbd', 'o')
    FTN.addLink('vd', 'r', 'qad', 'o')

    # left / right bonds
    FTN.addLink('wll', 'u', 'qbll', 'o')
    FTN.addLink('wll', 'd', 'qall', 'o')
    FTN.addLink('wrr', 'u', 'qbrr', 'o')
    FTN.addLink('wrr', 'd', 'qarr', 'o')

    # A left bonds
    FTN.addLink('aul', 'u', 'qbu', 'i')
    FTN.addLink('adl', 'd', 'qbd', 'i')
    FTN.addLink('aul', 'd', 'adl', 'u')
    FTN.addLink('aul', 'l', 'qbll', 'i')
    FTN.addLink('adl', 'l', 'qall', 'i')
    FTN.addLink('aul', 'r', 'qbrl', 'i')
    FTN.addLink('adl', 'r', 'qarl', 'i')

    # A right bonds
    FTN.addLink('aur', 'u', 'qau', 'i')
    FTN.addLink('adr', 'd', 'qad', 'i')
    FTN.addLink('aur', 'd', 'adr', 'u')
    FTN.addLink('aur', 'l', 'qblr', 'i')
    FTN.addLink('adr', 'l', 'qalr', 'i')
    FTN.addLink('aur', 'r', 'qbrr', 'i')
    FTN.addLink('adr', 'r', 'qarr', 'i')

    # center w bonds 
    FTN.addLink('wlr', 'u', 'qbrl', 'o')
    FTN.addLink('wlr', 'd', 'qarl', 'o')
    FTN.addLink('wrl', 'u', 'qblr', 'o')
    FTN.addLink('wrl', 'd', 'qalr', 'o')
    FTN.addLink('wlr', 'o', 'wrl', 'o')

    tensorDict = TensorDict()
    chi = 10
    for ATensorName in ATensorNames:
        tensorDict.setTensor(ATensorName, Tensor(data = np.random.randn(chi, chi, chi, chi), labels = ['u', 'l', 'd', 'r']))
    for QTensorName in QTensorNames:
        tensorDict.setTensor(QTensorName, Tensor(data = np.random.randn(chi, chi), labels = ['i', 'o']))
    for wTensorName in wTensorNames:
        tensorDict.setTensor(wTensorName, Tensor(data = np.random.randn(chi, chi, chi), labels = ['u', 'd', 'o']))
    for vTensorName in vTensorNames:
        tensorDict.setTensor(vTensorName, Tensor(data = np.random.randn(chi, chi, chi), labels = ['l', 'r', 'o']))
    
    res = FTN.contract(tensorDict)
    print(res)

def HOTRGGiltTNTruncated():
    ATensorNames = ["aul", "aur", "adl", "adr"]
    # QTensorNames = ['qbll', 'qbrl', 'qall', 'qarl', 'qblr', 'qbrr', 'qalr', 'qarr', 'qbu', 'qbd', 'qau', 'qad'] # i(circle), o(line)
    wTensorNames = ['wll', 'wlr', 'wrl', 'wrr'] # u, d, o
    vTensorNames = ['vu', 'vd'] # l, r, o

    tensorNames = ATensorNames + wTensorNames + vTensorNames

    FTN = FiniteTensorNetwork(tensorNames = tensorNames)

    # FTN.addLink('aul', 'adl', 'd', 'u')
    # FTN.addLink('aur', 'adr', 'd', 'u')

    # up / down bonds
    # FTN.addLink('vu', 'l', 'qbu', 'o')
    # FTN.addLink('vu', 'r', 'qau', 'o')
    # FTN.addLink('vd', 'l', 'qbd', 'o')
    # FTN.addLink('vd', 'r', 'qad', 'o')

    # left / right bonds
    # FTN.addLink('wll', 'u', 'qbll', 'o')
    # FTN.addLink('wll', 'd', 'qall', 'o')
    # FTN.addLink('wrr', 'u', 'qbrr', 'o')
    # FTN.addLink('wrr', 'd', 'qarr', 'o')

    # A left bonds
    FTN.addLink('aul', 'u', 'vu', 'l')
    FTN.addLink('adl', 'd', 'vd', 'l')
    FTN.addLink('aul', 'd', 'adl', 'u')
    FTN.addLink('aul', 'l', 'wll', 'u')
    FTN.addLink('adl', 'l', 'wll', 'd')
    FTN.addLink('aul', 'r', 'wlr', 'u')
    FTN.addLink('adl', 'r', 'wlr', 'd')

    # A right bonds
    FTN.addLink('aur', 'u', 'vu', 'r')
    FTN.addLink('adr', 'd', 'vd', 'r')
    FTN.addLink('aur', 'd', 'adr', 'u')
    FTN.addLink('aur', 'l', 'wrl', 'u')
    FTN.addLink('adr', 'l', 'wrl', 'd')
    FTN.addLink('aur', 'r', 'wrr', 'u')
    FTN.addLink('adr', 'r', 'wrr', 'd')

    # center w bonds 
    # FTN.addLink('wlr', 'u', 'qbrl', 'o')
    # FTN.addLink('wlr', 'd', 'qarl', 'o')
    # FTN.addLink('wrl', 'u', 'qblr', 'o')
    # FTN.addLink('wrl', 'd', 'qalr', 'o')
    FTN.addLink('wlr', 'o', 'wrl', 'o')

    tensorDict = TensorDict()
    chi = 10
    for ATensorName in ATensorNames:
        tensorDict.setTensor(ATensorName, Tensor(data = np.random.randn(chi, chi, chi, chi), labels = ['u', 'l', 'd', 'r']))
    # for QTensorName in QTensorNames:
        # tensorDict.setTensor(QTensorName, Tensor(data = np.random.randn(chi, chi), labels = ['i', 'o']))
    for wTensorName in wTensorNames:
        tensorDict.setTensor(wTensorName, Tensor(data = np.random.randn(chi, chi, chi), labels = ['u', 'd', 'o']))
    for vTensorName in vTensorNames:
        tensorDict.setTensor(vTensorName, Tensor(data = np.random.randn(chi, chi, chi), labels = ['l', 'r', 'o']))
    
    res = FTN.contract(tensorDict)
    print(res)

if __name__ == '__main__':
    HOTRGGiltTNTruncated() # 0.44s
    # HOTRGGiltTN() # 78.79s
    

