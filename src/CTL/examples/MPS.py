# calculation of MPS
# given a tensor, return the MPS by taking each bond as an MPS tensor

import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contract import shareBonds, contractTwoTensors
from CTL.tensor.contract.optimalContract import copyTensorList
from CTL.examples.Schimdt import SchimdtDecomposition
from CTL.tensor.tensorFunc import isIsometry
import warnings

class FreeBoundaryMPS:

    '''
    MPS:
    Maintain a list of tensors, the first & last tensors are 2-dimensional, others are 3-dimensional
    The outer bonds are just the legs of the input tensor, and the internal bonds are with bond dimension at max chi
    Support tensor swap?
    '''

    def checkMPSProperty(self, tensorList):
        # MPS property: first and last tensor has one bond out, and another linked to the next
        # others: one to left, one to right, one out

        n = len(tensorList)

        if (n == 0):
            return False 

        for i in range(n - 1):
            if (len(shareBonds(tensorList[i], tensorList[i + 1])) != 1):
                return False 
        
        if (tensorList[0].dim != 2) or (tensorList[-1].dim != 2):
            return False
        
        for i in range(1, n - 1):
            if (tensorList[i].dim != 3):
                return False 
        
        return True

    def getChi(self, chi):
        # if chi is None: then take the maximum from bonds shared
        # otherwise, if bonds sharing larger than chi, then warning and update chi
        # otherwise, take chi
        bondChi = -1
        for i in range(self.n - 1):
            bond = shareBonds(self.tensors[i], self.tensors[i + 1])[0]
            bondChi = min(bondChi, bond.legs[0].dim) 
        
        if (chi is None):
            return bondChi 
        elif (bondChi > chi):
            warnings.warn(funcs.warningMessage('required chi {} is lower than real bond chi {}, set to {}.'.format(chi, bondChi, bondChi), location = "FreeBoundaryMPS.getChi"))
            return bondChi 
        else:
            return chi

    def renameBonds(self):
        # 'l' and 'r' for internal bonds
        # 'o' for external bonds
        self.internalBonds = set()
        for i in range(self.n - 1):
            bond = shareBonds(self.tensors[i], self.tensors[i + 1])[0]
            bond.sideLeg(self.tensors[i]).name = 'r'
            bond.sideLeg(self.tensors[i + 1]).name = 'l'
            self.internalBonds.add(bond)

        for i in range(self.n):
            for leg in self.tensors[i].legs:
                if (leg.bond not in self.internalBonds):
                    leg.name = 'o'

    def __init__(self, tensorList, chi = None):
        if (not self.checkMPSProperty(tensorList)):
            raise ValueError(funcs.errorMessage("tensorList {} cannot be considered as an MPS".format(tensorList), location = "FreeBoundaryMPS.__init__"))
        
        self.tensors = copyTensorList(tensorList)
        self.n = len(self.tensors)

        self.chi = self.getChi(chi)
        self.renameBonds()
    
    def __repr__(self):
        return 'FreeBoundaryMPS(tensors = {}, chi = {})'.format(self.tensors, self.chi)

    def canonicalize(self, direct = 0):
        # the workflow of canonicalization:
        # direct = 0: left to right
        # direct = 1: right to left
        # consider direct = 0: first contract 0 and 1
        # then do an SVD over o0 and (o1, r1): T_0 T_1 = U S V
        # the bond dimension should be min(o0, (o1, r1), chi)
        # then we take U as new tensors[0], and SV to tensors[1]
        # do the same for 1, 2, ... n - 2, and final S is into tensors[n - 1

        assert (direct in [0, 1]), funcs.errorMessage("direct must in [0, 1].", location = "FreeBoundaryMPS.canonicalize")
        
        if (direct == 0):
            for i in range(self.n - 1):
                u, s, v = SchimdtDecomposition(self.tensors[i], self.tensors[i + 1], self.chi)
                sv = contractTwoTensors(s, v) 
                self.tensors[i] = u
                self.tensors[i + 1] = sv
        else:
            for i in range(self.n - 1, 0, -1):
                u, s, v = SchimdtDecomposition(self.tensors[i], self.tensors[i - 1], self.chi)
                sv = contractTwoTensors(s, v) 
                self.tensors[i] = u
                self.tensors[i - 1] = sv

    def swap(self, aIdx, bIdx):
        # tensorA and tensorB are tensors in tensorList
        assert ((aIdx >= 0) and (aIdx < self.n) and (bIdx < self.n) and (bIdx >= 0) and (abs(aIdx - bIdx) == 1)), funcs.errorMessage("index {} and {} are not valid for MPS with {} tensors.".format(aIdx, bIdx, self.n), location = "FreeBoundaryMPS.swap")
        
        self.tensors[aIdx], _, self.tensors[bIdx] = SchimdtDecomposition(self.tensors[aIdx], self.tensors[bIdx], self.chi, squareRootSeparation = True, swapLabels = (['o'], ['o']))

    def checkCanonical(self, direct = 0):
        '''
        check if the current MPS is in canonical 
        '''
        funcName = 'FreeBoundaryMPS.checkCanonical'
        assert (direct in [0, 1]), funcs.errorMessage("direct must in [0, 1].", location = funcName)
        if (self.n == 0):
            warnings.warn(funcs.warningMessage("number of tensors in MPS is 0, return True", location = funcName))
            return True
        if (direct == 0):
            if (not isIsometry(self.tensors[0], ['o'])):
                return False 
            for i in range(1, self.n - 1):
                if (not isIsometry(self.tensors[i], ['l', 'o'])):
                    return False 
            return True 
        else:
            if (not isIsometry(self.tensors[-1], ['o'])):
                return False 
            for i in range(self.n - 2, 0, -1):
                if (not isIsometry(self.tensors[i], ['r', 'o'])):
                    return False 
            return True
        
