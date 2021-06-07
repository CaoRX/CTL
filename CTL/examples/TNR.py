# -*- coding: utf-8 -*-
# doTNR.py

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from numpy import linalg as LA
from ncon import ncon

from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.contractExp import EvenblyTNRQEnvFTN, selfTrace
from CTL.tensornetwork.tensornetwork import FiniteTensorNetwork
import CTL.funcs.funcs as funcs 
from CTL.tensornetwork.tensordict import TensorDict
from CTL.funcs.linalg import solveEnv

def createQxAFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['q', 'A'])

    FTN.addLink('q', 'h', 'A', 'l')
    FTN.addLink('q', 'v', 'A', 'u')

    FTN.addPostNameChange('A', 'r', 'h')
    FTN.addPostNameChange('A', 'd', 'v')

    return FTN

def createCFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['ul', 'ur', 'dl', 'dr'])

    FTN.addLink('ul', 'v', 'dl', 'v')
    FTN.addLink('ur', 'v', 'dr', 'v')
    FTN.addLink('ul', 'h', 'ur', 'h')
    FTN.addLink('dl', 'h', 'dr', 'h')

    for name in ['ul', 'ur', 'dl', 'dr']:
        FTN.addPostNameChange(name, 'o', name)

    return FTN

def createCCFTN():
    FTN = FiniteTensorNetwork(tensorNames = ['u', 'd'])
    
    FTN.addLink('u', 'dl', 'd', 'dl')
    FTN.addLink('u', 'dr', 'd', 'dr')
    
    FTN.addPostNameChange('d', 'ul', 'dl')
    FTN.addPostNameChange('d', 'ur', 'dr')

    return FTN

    # Cdub = ncon([C,C],[[-1,-2,1,2],[-3,-4,1,2]])
    # C = CTensor.toTensor(labels = ['ul', 'ur', 'dl', 'dr'])

def doTNR(A, allchi, dtol = 1e-10, disiter = 2000, miniter = 100, dispon = True, convtol = 0.01):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 29/1/2019
------------------------
Implementation of TNR using implicit disentangling. Input 'A' is a four \
index tensor that defines the (square-lattice) partition function while \
'allchi = [chiM,chiS,chiU,chiH,chiV]' are the bond dimensions.

Optional arguments:
`dtol::Float=1e-10`: threshold for discarding small eigenvalues
`disiter::Int=2000`: max iterations in disentangler optimization
`miniter::Int=100`: min iterations in disentangler optimization
`dispon::Bool=true`: display information during optimization
`convtol::Float=1e-2`: halt optimization if changes smaller than convtol 
"""
    chiHI = A.shape[0]
    chiVI = A.shape[1]
    chiM = min(allchi[0], chiHI*chiVI)
    chiU = min(allchi[2], chiVI)
    chiH = min(allchi[3], chiHI**2)
    chiV = min(allchi[4], chiU**2)

    ##### determine 'qM' isometry
    aTensor = Tensor(data = A, labels = ['l', 'u', 'r', 'd'])
    qEnvFTN = EvenblyTNRQEnvFTN()
    qEnvTensorDict = TensorDict(funcs.identicalTensorDict(aTensor, ['uul', 'uur', 'udl', 'udr', 'dul', 'dur', 'ddl', 'ddr']))
    qEnv = qEnvFTN.contract(qEnvTensorDict).toMatrix(rows = ['1'], cols = ['2'])
    qTemp, _ = solveEnv(qEnv, chi = chiM)

    qM = qTemp.reshape(chiHI, chiVI, qTemp.shape[1])
    qTensor = Tensor(data = qTemp, labels = ['h', 'v', 'o'], shape = (chiHI, chiVI, qTemp.shape[1]))

    chiM = qTemp.shape[1]
    chiS = min(allchi[1], chiM)

    SP1exact = np.trace(qEnv)
    SP1err = abs((SP1exact - np.trace(qTemp.T @ qEnv @ qTemp)) / SP1exact) 

    qxAFTN = createQxAFTN()
    qATensor = qxAFTN.contract({'A': aTensor, 'q': qTensor})
    qA = qATensor.toTensor(labels = ['v', 'h', 'o'])
    # qA = ncon([qM,A],[[1,2,-3],[1,2,-2,-1]])
    cFTN = createCFTN()
    # C = ncon([qA,qA,qA,qA],[[1,3,-1],[2,3,-2],[1,4,-3],[2,4,-4]])
    CTensor = cFTN.contract(funcs.identicalTensorDict(qATensor, ['ul', 'ur', 'dl', 'dr']))
    C = CTensor.toTensor(labels = ['ul', 'ur', 'dl', 'dr'])

    ###### iteration to determine 'sM' matrix, 'yM' isometry, 'uM' disentangler
    uM = np.kron(np.eye(chiVI,chiU),np.eye(chiVI,chiU)).reshape(chiVI,chiVI,chiU,chiU)
    uTensor = Tensor(data = uM, labels = ['ol', 'or', 'il', 'ir']) # assume i direction is upward
    yM = qM[:, :uM.shape[2], :chiS]
    yTensor = Tensor(data = yM, labels = ['h', 'v', 'o'])
    sM = np.eye(qM.shape[2],chiS)
    sTensor = Tensor(data = sM, labels = ['i', 'o'])

    CCFTN = createCCFTN()
    CdubTensor = CCFTN.contract(funcs.identicalTensorDict(CTensor, ['u', 'd']))
    # Cdub = ncon([C,C],[[-1,-2,1,2],[-3,-4,1,2]])
    Cdub = CdubTensor.toTensor(labels = ['ul', 'ur', 'dl', 'dr'])
    sCenvD = Cdub.transpose(0,2,1,3)
    # SP2exact = ncon([C,C],[[1,2,3,4],[1,2,3,4]])
    SP2exact = selfTrace(CTensor).single()
    SP2err = 1

    for k in range(disiter + 1):
        sCenvS = ncon([Cdub,qM,qM,uM,yM,yM],[[-1,-3,7,8],[1,3,7],[4,6,8],[3,6,2,5],[1,2,-2],[4,5,-4]])
        senvS = ncon([sCenvS,sM],[[-1,-2,1,2],[1,2]])
        senvD = ncon([sCenvD,sM @ (sM.T)],[[-1,-2,1,2],[1,2]])

        if np.mod(k,100) == 0:
            SP2errnew = abs(1 - (np.trace(senvS @ (sM.T))**2) / (np.trace((sM.T) @ senvD @ sM)*SP2exact)) 
            if k > 50:
                errdelta = abs(SP2errnew-SP2err) / abs(SP2errnew)
                if (errdelta < convtol) or (abs(SP2errnew) < 1e-10):
                    SP2err = SP2errnew
                    if dispon:
                        print('Iteration: %d of %d, Trunc. Error: %e, %e' % (k,disiter,SP1err,SP2err))
                    break
                
            SP2err = SP2errnew
            if dispon:
                print('Iteration: %d of %d, Trunc. Error: %e, %e' % (k,disiter,SP1err,SP2err))
            
        #     stemp = senvD\senvS;
        stemp = LA.pinv(senvD/np.trace(senvD),rcond = dtol) @ senvS
        stemp = stemp/LA.norm(stemp)

        Serrold = abs(1-(np.trace(senvS @ (sM.T))**2) / (np.trace((sM.T) @ senvD @ sM)*SP2exact))
        for p in range(11):
            snew = (1 - 0.1*p)*stemp + 0.1*p*sM
            Serrnew = abs(1 - (ncon([sCenvS,snew,snew],[[1,2,3,4],[1,2],[3,4]])**2)/
                (ncon([sCenvD,snew @ (snew.T), snew @ (snew.T)],[[1,2,3,4],[1,2],[3,4]])*SP2exact))
            if Serrnew <= Serrold:
                sM = snew/LA.norm(snew)
                break
            
        if k > 50:
            yenv = ncon([C,qM,qM,uM,yM,sM,sM,C],[[10,6,3,4],[-1,11,10],[5,8,6],
                         [11,8,-2,9],[5,9,7],[1,-3],[2,7],[1,2,3,4]])
            yM = TensorUpdateSVD(yenv,2)

            uenv = ncon([C,qM,qM,yM,yM,sM,sM,C],[[6,9,3,4],[5,-1,6],[8,-2,9],
                         [5,-3,7],[8,-4,10],[1,7],[2,10],[1,2,3,4]])
            uenv = uenv + uenv.transpose(1,0,3,2)
            uM = TensorUpdateSVD(uenv,2)
    
    Cmod = ncon([C,sM,sM,sM,sM],[[1,2,3,4],[1,-1],[2,-2],[3,-3],[4,-4]])
    Cnorm = ncon([Cmod,Cmod],[[1,2,3,4],[1,2,3,4]]) / ncon([C,C],[[1,2,3,4],[1,2,3,4]])
    sM = sM / (Cnorm**(1/8)) 

    ###### determine 'vM' isometry
    venv = ncon([yM,yM,yM,yM,sM,qA,qA,sM,sM,qA,qA,sM,yM,yM,yM,yM,sM,qA,qA,sM,sM,qA,qA,sM],
                [[1,3,17],[1,4,24],[2,3,18],[2,4,29],[5,17],[7,11,5],[7,12,6],[6,19],
                 [8,18],[10,11,8],[10,12,9],[9,20],[13,15,19],[13,16,25],[14,15,20],[14,16,30],
                 [21,24],[23,-1,21],[23,-2,22],[22,25],[26,29],[28,-3,26],[28,-4,27],[27,30]])
    venv = 0.5*(venv + venv.transpose(1,0,3,2)).reshape(chiHI**2,chiHI**2)
    vtemp, _ = solveEnv(venv, chi = chiH)
    vM = vtemp.reshape(chiHI,chiHI,vtemp.shape[1])

    SP3exact = np.trace(venv)
    SP3err = abs((SP3exact - np.trace(vtemp.T @ venv @ vtemp))/SP3exact)

    ###### determine 'wM' isometry
    wenv = ncon([yM,yM,yM,yM,sM,qA,qA,sM,sM,qA,qA,sM,yM,yM,yM,yM,sM,qA,qA,sM,sM,qA,qA,sM],
                [[25,-1,26],[25,-2,27],[28,-3,29],[28,-4,30],[1,26],[3,7,1],[3,8,2],[2,13],
                 [4,29],[6,7,4],[6,8,5],[5,14],[9,11,13],[9,12,23],[10,11,14],[10,12,24],
                 [15,27],[17,21,15],[17,22,16],[16,23],[18,30],[20,21,18],[20,22,19],[19,24]])
    wenv = 0.5*(wenv + wenv.transpose(1,0,3,2)).reshape(chiU**2,chiU**2)
    wtemp, _ = solveEnv(wenv, chi = chiV)
    wM = wtemp.reshape(chiU,chiU,wtemp.shape[1])

    SP4exact = np.trace(wenv)
    SP4err = abs((SP4exact - np.trace(wtemp.T @ wenv @ wtemp))/SP4exact)

    ###### generate new 'A' tensor
    Atemp = ncon([vM,sM,qA,qA,sM,wM,yM,yM,vM,sM,qA,qA,sM,wM,yM,yM],
                 [[10,9,-1],[7,19],[6,9,7],[6,10,8],[8,14],[17,18,-2],[16,17,19],[16,18,20],
                  [4,5,-3],[1,20],[3,4,1],[3,5,2],[2,15],[13,12,-4],[11,12,14],[11,13,15]])
    Anorm = LA.norm(Atemp)
    Aout = Atemp / Anorm

    SPerrs = np.array([SP1err,SP2err,SP3err,SP4err])

    return Aout, qM, sM, uM, yM, vM, wM, Anorm, SPerrs


"""
TensorUpdateSVD: update an isometry or unitary tensor using its \
(linearized) environment
"""
def TensorUpdateSVD(wIn,leftnum):

    wSh = wIn.shape
    ut,st,vht = LA.svd(wIn.reshape(np.prod(wSh[0:leftnum:1]),np.prod(wSh[leftnum:len(wSh):1])),full_matrices=False)
    return (ut @ vht).reshape(wSh)
