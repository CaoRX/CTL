import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contractExp import CTMRGHEdgeExtendFTN, CTMRGVEdgeExtendFTN, CTMRGCornerExtendFTN, CTMRGHEdgeBuildFTN, CTMRGVEdgeBuildFTN, CTMRGEvenZFTN, CTMRGOddFTN
from CTL.tensor.tensorFunc import tensorSVDDecomposition
from CTL.tensor.contract.contract import contractTwoTensors, makeLink

# import numpy as np
import CTL.funcs.xplib as xplib
import warnings
class CTMRG:
    # initial tensor: corner, edge, center
    # center will always be self.a
    # if only given center: sum out 1

    def setInitialTensors(self):
        self.tensors = dict()
        self.tensors['corner'], self.tensors['hEdge'], self.tensors['vEdge'] = self.a.copyN(3)
        self.tensors['corner'].sumOutLegByLabel(['d', 'l'])
        self.tensors['hEdge'].sumOutLegByLabel('d')
        self.tensors['vEdge'].sumOutLegByLabel('l')

        self.logScale = dict()
        for tensorName in ['corner', 'hEdge', 'vEdge']:
            self.logScale[tensorName] = 0.0
        # the tensor of <name> is normalized by a factor exp(logScale[name])
        self.tensors['corner'], self.tensors['hEdge'], self.tensors['vEdge'], error = self.reduceDimension(self.tensors['corner'], self.tensors['hEdge'], self.tensors['vEdge'])
        self.errors.append(error)

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
        self.vBuildFTN = CTMRGVEdgeBuildFTN()
        self.hBuildFTN = CTMRGHEdgeBuildFTN()

        self.evenZFTN = CTMRGEvenZFTN()
        self.oddZFTN = CTMRGOddFTN()
    
    def __init__(self, a, chi = 16):
        self.a = a.copy()
        self.chi = chi

        assert funcs.compareLists(a.labels, ['l', 'r', 'u', 'd']), funcs.errorMessage('CTMRG can only accept tensor with legs ["l", "r", "u", "d"], {} obtained.'.format(a))

        self.setRecords()
        self.setFTNs()
        self.setInitialTensors()

    def reduceDimension(self, corner, hEdge, vEdge):
        # TODO: implement the algorithm to make bond dimension lower than chi
        # return corner, hEdge, vEdge, 0.0

        # what should be returned:
        # corner: diagonal tensor, bond dimesion D = max(d, chi)
        # hEdge, vEdge: rank-3 tensor, (D, D, a)

        # print('reduceDimension(corner = {}, hEdge = {}, vEdge = {})'.format(corner, hEdge, vEdge))

        decomp = tensorSVDDecomposition(corner, rows = ['u'], cols = ['r'], innerLabels = ('d', 'l'), chi = self.chi, errorOrder = 4)

        # print(decomp)
        error = decomp['error']
        # print('error = {}'.format(error))

        newCorner = decomp['s']
        # print('corner.trace = {}, newCorner.trace = {}'.format(corner.trace(rows = ['u'], cols = ['r']), newCorner.trace()))

        vTensorUp, vTensorDown = decomp['u'].copyN(2)
        hTensorLeft, hTensorRight = decomp['v'].copyN(2)

        newVEdge = self.vBuildFTN.contract({'au': vTensorUp, 'ad': vTensorDown, 'p': vEdge})
        newHEdge = self.hBuildFTN.contract({'al': hTensorLeft, 'ar': hTensorRight, 'p': hEdge})

        h = newHEdge.copy()
        h.reArrange(['u', 'l', 'r'])
        # h.sumOutLegByLabel('u')
        # print('horizontal edge tensor = {}'.format(h.a))
        return newCorner, newHEdge, newVEdge, error

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
        logScale['corner'] = xplib.xp.log(cornerNorm) + self.logScale['corner'] + self.logScale['hEdge'] + self.logScale['vEdge']
        logScale['vEdge'] = xplib.xp.log(vEdgeNorm) + self.logScale['vEdge']
        logScale['hEdge'] = xplib.xp.log(hEdgeNorm) + self.logScale['hEdge']

        self.logScale = logScale
        self.tensors['corner'] = newCorner
        self.tensors['vEdge'] = newVEdge
        self.tensors['hEdge'] = newHEdge

        self.errors.append(error)

    # def getSingleZ(self, L):
    #     funcName = 'CTMRG.getSingleZ'
    #     warnings.warn(funcs.warningMessage("This function may not be a proper way to calculate partition function of CTMRG, please use getZ(L = <odd-int>) instead.", location = funcName), RuntimeWarning)
    #     # L = 1: self-trace of initial corner
    #     while(len(self.records['corner']) < L - 1):
    #         self.iterate()
    #     if (len(self.records['corner']) == L - 1):
    #         # then current tensor is L * L tensor
    #         corner = self.tensors['corner'].copy()
    #         logZ = self.logScale['corner']
    #     else:
    #         corner, logZ = self.records['corner'][L - 1]
    #         corner = corner.copy()
    #     print(corner)
    #     print('corner.data = {}'.format(corner.a))
    #     corner.sumOutLegByLabel(['u', 'r'])
    #     print(corner)
    #     return corner.single() * xplib.xp.exp(logZ)

    def getTensorAndScale(self, name, idx):
        # print(idx, len(self.records[name]))
        resTensor, resScale = self.records[name][idx]
        return resTensor.copy(), resScale

    def getZ(self, L):
        funcName = 'CTMRG.getZ'

        if (L == 1):
            a = self.a.copy()
            a.sumOutLegByLabel(['l', 'r', 'u', 'd'])
            return a.single()

        minimumOrder = L // 2
        while (len(self.records['corner']) < minimumOrder - 1):
            self.iterate()
        if (len(self.records['corner']) == minimumOrder - 1):
            corner = self.tensors['corner'].copy()
            cornerLogZ = self.logScale['corner']
            if (L % 2 == 1):
                hEdge = self.tensors['hEdge'].copy()
                hEdgeLogZ = self.logScale['hEdge']
                vEdge = self.tensors['vEdge'].copy()
                vEdgeLogZ = self.logScale['vEdge']
        else:
            idx = minimumOrder - 1
            corner, cornerLogZ = self.getTensorAndScale('corner', idx)
            if (L % 2 == 1):
                hEdge, hEdgeLogZ = self.getTensorAndScale('hEdge', idx)
                vEdge, vEdgeLogZ = self.getTensorAndScale('vEdge', idx)

        tensors = dict()
        tensors['alu'] = tensors['ald'] = tensors['aru'] = tensors['ard'] = corner
        if (L % 2 == 1):
            tensors['pl'] = tensors['pr'] = vEdge 
            tensors['pd'] = tensors['pu'] = hEdge 
            tensors['w'] = self.a
        # print(tensors)
        if (L % 2 == 0):
            res = self.evenZFTN.contract(tensors)
            # print(res)
            return res.single() * xplib.xp.exp(4 * cornerLogZ)
        else:
            res = self.oddZFTN.contract(tensors)
            return res.single() * xplib.xp.exp(4 * cornerLogZ + 2 * hEdgeLogZ + 2 * vEdgeLogZ)
    

        

        
