import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contractExp import CTMRGHEdgeExtendFTN, CTMRGVEdgeExtendFTN, CTMRGCornerExtendFTN

import numpy as np
import warnings
class CTMRG:
    # initial tensor: corner, edge, center
    # center will always be self.a
    # if only given center: sum out 1

    def setInitialTensors(self, chi = 16):
        self.tensors = dict()
        self.tensors['corner'], self.tensors['hEdge'], self.tensors['vEdge'] = self.a.copyN(3)
        self.tensors['corner'].sumOutLegByLabel(['d', 'l'])
        self.tensors['hEdge'].sumOutLegByLabel('d')
        self.tensors['vEdge'].sumOutLegByLabel('l')

        self.logScale = dict()
        for tensorName in ['corner', 'hEdge', 'vEdge']:
            self.logScale[tensorName] = 0.0
        # the tensor of <name> is normalized by a factor exp(logScale[name])

        self.chi = chi

    def setRecords(self):
        self.records = dict()
        for tensorName in ['corner', 'hEdge', 'vEdge']:
            self.records[tensorName] = []

        self.errors = []

    def pushTensors(self):
        for tensorName in self.tensors:
            self.records[tensorName].append((self.tensors[tensorName].copy(), self.logScale[tensorName]))

    def setFTNs(self):
        self.hEdgeExtendFTN = CTMRGHEdgeExtendFTN()
        self.vEdgeExtendFTN = CTMRGVEdgeExtendFTN()
        self.cornerExtendFTN = CTMRGCornerExtendFTN()
    
    def __init__(self, a, chi = 16):
        self.a = a.copy()

        assert funcs.compareLists(a.labels, ['l', 'r', 'u', 'd']), funcs.errorMessage('CTMRG can only accept tensor with legs ["l", "r", "u", "d"], {} obtained.'.format(a))

        self.setInitialTensors()
        self.setRecords()
        self.setFTNs()

    def reduceDimension(self, corner, hEdge, vEdge):
        # TODO: implement the algorithm to make bond dimension lower than chi
        return corner, hEdge, vEdge, 0.0

    def iterate(self):
        self.pushTensors()

        corner = self.tensors['corner'].copy()
        hEdge = self.tensors['hEdge'].copy()
        vEdge = self.tensors['vEdge'].copy()

        newHEdge = self.hEdgeExtendFTN.contract({'p': hEdge, 'w': self.a})
        newVEdge = self.vEdgeExtendFTN.contract({'p': vEdge, 'w': self.a})
        newCorner = self.cornerExtendFTN.contract({'ph': hEdge, 'pv': vEdge, 'w': self.a, 'c': corner})

        newCorner, newHEdge, newVEdge, error = self.reduceDimension(newCorner, newHEdge, newVEdge)

        cornerNorm = newCorner.norm()
        hEdgeNorm = newHEdge.norm()
        vEdgeNorm = newVEdge.norm()

        newCorner.a /= cornerNorm
        newHEdge.a /= hEdgeNorm
        newVEdge.a /= vEdgeNorm

        logScale = dict()
        logScale['corner'] = np.log(cornerNorm) + self.logScale['corner'] + self.logScale['hEdge'] + self.logScale['vEdge']
        logScale['vEdge'] = np.log(vEdgeNorm) + self.logScale['vEdge']
        logScale['hEdge'] = np.log(hEdgeNorm) + self.logScale['hEdge']

        self.logScale = logScale
        self.tensors['corner'] = newCorner
        self.tensors['vEdge'] = newVEdge
        self.tensors['hEdge'] = newHEdge

        self.errors.append(error)

    def getSingleZ(self, L):
        funcName = 'CTMRG.getSingleZ'
        warnings.warn(funcs.warningMessage("This function may not be a proper way to calculate partition function of CTMRG, please use getZ(L = <odd-int>) instead.", location = funcName), RuntimeWarning)
        # L = 1: self-trace of initial corner
        while(len(self.records['corner']) < L - 1):
            self.iterate()
        if (len(self.records['corner']) == L - 1):
            # then current tensor is L * L tensor
            corner = self.tensors['corner'].copy()
            logZ = self.logScale['corner']
        else:
            corner, logZ = self.records['corner'][L - 1]

        corner.sumOutLegByLabel(['u', 'r'])
        return corner.single() * np.exp(logZ)

    

        

        
