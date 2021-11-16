# calculation of MPS
# given a tensor, return the MPS by taking each bond as an MPS tensor

import CTL.funcs.xplib as xplib
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contract import shareBonds, contractTwoTensors, merge
from CTL.tensor.tensor import Tensor
from CTL.tensor.leg import Leg
from CTL.tensor.contract.optimalContract import copyTensorList, contractWithSequence, generateOptimalSequence, generateGreedySequence
from CTL.examples.Schimdt import SchimdtDecomposition, matrixSchimdtDecomposition
from CTL.tensor.tensorFunc import isIsometry
import warnings
from CTL.tensor.contract.link import makeLink

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
        
        if (n > 1) and ((tensorList[0].dim != 2) or (tensorList[-1].dim != 2)):
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
            bond = shareBonds(self._tensors[i], self._tensors[i + 1])[0]
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
            bond = shareBonds(self._tensors[i], self._tensors[i + 1])[0]
            bond.sideLeg(self._tensors[i]).name = 'r'
            bond.sideLeg(self._tensors[i + 1]).name = 'l'
            self.internalBonds.add(bond)

        for i in range(self.n):
            for leg in self._tensors[i].legs:
                if (leg.bond not in self.internalBonds):
                    leg.name = 'o'

    def __init__(self, tensorList, chi = None, inplace = True):
        if (not self.checkMPSProperty(tensorList)):
            raise ValueError(funcs.errorMessage("tensorList {} cannot be considered as an MPS".format(tensorList), location = "FreeBoundaryMPS.__init__"))
        
        if (inplace):
            self._tensors = tensorList 
        else:
            self._tensors = copyTensorList(tensorList, linkOutgoingBonds = True)
        # self._tensors = copyTensorList(tensorList)
        # _tensors should not be modified directly: it will destroy the property self.activeIdx
        # please use MPS.setTensor(idx, tensor) for changing the tensors

        self.chi = self.getChi(chi)
        self.renameBonds()

        self.activeIdx = None
    
    @property 
    def n(self):
        return len(self._tensors)
    def __repr__(self):
        return 'FreeBoundaryMPS(tensors = {}, chi = {})'.format(self._tensors, self.chi)

    # def canonicalize(self, direct = 0):
    #     # the workflow of canonicalization:
    #     # direct = 0: left to right
    #     # direct = 1: right to left
    #     # consider direct = 0: first contract 0 and 1
    #     # then do an SVD over o0 and (o1, r1): T_0 T_1 = U S V
    #     # the bond dimension should be min(o0, (o1, r1), chi)
    #     # then we take U as new tensors[0], and SV to tensors[1]
    #     # do the same for 1, 2, ... n - 2, and final S is into tensors[n - 1

    #     assert (direct in [0, 1]), funcs.errorMessage("direct must in [0, 1].", location = "FreeBoundaryMPS.canonicalize")
        
    #     if (direct == 0):
    #         for i in range(self.n - 1):
    #             u, s, v = SchimdtDecomposition(self._tensors[i], self._tensors[i + 1], self.chi)
    #             sv = s @ v 
    #             self._tensors[i] = u
    #             self._tensors[i + 1] = sv
    #     else:
    #         for i in range(self.n - 1, 0, -1):
    #             u, s, v = SchimdtDecomposition(self._tensors[i], self._tensors[i - 1], self.chi)
    #             sv = s @ v 
    #             self._tensors[i] = u
    #             self._tensors[i - 1] = sv

    def canonicalize(self, idx):
        '''
        canonicalize the MPS, and the only non-isometry will be put at 0 <= idx < n
        after this, activeIdx will be set to idx
        and we can check the canonicalization of the MPS with self.checkCanonical
        if None for excepIdx(or set to the idx), it will give true before modified
        '''
        assert (isinstance(idx, int) and (idx >= 0) and (idx < self.n)), funcs.errorMessage('index must in [0, n - 1), {} gotten.'.format(idx), location = "FreeBoundaryMPS.canonicalize")
        for i in range(idx):
            # print('canonicalizing {} to {}'.format(i, i + 1))
            u, s, v = SchimdtDecomposition(self._tensors[i], self._tensors[i + 1], self.chi)
            sv = contractTwoTensors(s, v)
            self._tensors[i] = u
            self._tensors[i + 1] = sv 
        for i in range(self.n - 1, idx, -1):
            # print('canonicalizing {} to {}'.format(i, i - 1))
            u, s, v = SchimdtDecomposition(self._tensors[i], self._tensors[i - 1], self.chi)
            # print('isometry = {}'.format(isIsometry(u, ['o'])))
            sv = contractTwoTensors(s, v)
            self._tensors[i] = u
            self._tensors[i - 1] = sv

        self.activeIdx = idx

    def setTensor(self, idx, tensor):
        assert (idx >= 0 and idx < self.n), funcs.errorMessage("index must be in [0, n) but {} obtained.".format(idx), location = "FreeBoundaryMPS.setTensor")
        self._tensors[idx] = tensor 
        self.activeIdx = None
    def getTensor(self, idx):
        return self._tensors[idx]

    def swap(self, aIdx, bIdx):
        # tensorA and tensorB are tensors in tensorList
        assert ((aIdx >= 0) and (aIdx < self.n) and (bIdx < self.n) and (bIdx >= 0) and (abs(aIdx - bIdx) == 1)), funcs.errorMessage("index {} and {} are not valid for MPS with {} tensors.".format(aIdx, bIdx, self.n), location = "FreeBoundaryMPS.swap")
        
        self._tensors[aIdx], _, self._tensors[bIdx] = SchimdtDecomposition(self._tensors[aIdx], self._tensors[bIdx], self.chi, squareRootSeparation = True, swapLabels = (['o'], ['o']))
        
        self.activeIdx = None

    def checkCanonical(self, excepIdx = None):
        '''
        check if the current MPS is in canonical(isometry except for excepIdx)
        if the index is not given: check with the index the last time the MPS has been canonicalized for
        !!! Note that: this will be not true when excepIdx is None if the MPS has been changed directly by self._tensors[...] = ...
        '''
        funcName = 'FreeBoundaryMPS.checkCanonical'
        assert (excepIdx is not None) or (self.activeIdx is not None), funcs.errorMessage("exception index and MPS.activeIdx cannot be None at the same time.", location = funcName)
        if (excepIdx is None):
            excepIdx = self.activeIdx
        assert (isinstance(excepIdx, int) and (excepIdx >= 0) and (excepIdx < self.n)), funcs.errorMessage("exception index must in [0, self.n), {} obtained.".format(excepIdx), location = funcName)
        if (self.n == 0):
            warnings.warn(funcs.warningMessage("number of tensors in MPS is 0, return True", location = funcName))
            return True
        # print([isIsometry(tensor, ['o']) for tensor in self._tensors])
        for i in range(self.n):
            if (i == excepIdx):
                continue 
            if (i == 0) or (i == self.n - 1):
                labels = ['o']
            elif (i < excepIdx):
                labels = ['l', 'o']
            else:
                labels = ['r', 'o']
            # print(i, labels, isIsometry(self._tensors[i], labels))
            if not isIsometry(self._tensors[i], labels):
                return False 
        
        return True
        # if (direct == 0):
        #     if (not isIsometry(self._tensors[0], ['o'])):
        #         return False 
        #     for i in range(1, self.n - 1):
        #         if (not isIsometry(self._tensors[i], ['l', 'o'])):
        #             return False 
        #     return True 
        # else:
        #     if (not isIsometry(self._tensors[-1], ['o'])):
        #         return False 
        #     for i in range(self.n - 2, 0, -1):
        #         if (not isIsometry(self._tensors[i], ['r', 'o'])):
        #             return False 
        #     return True
    def isIndex(self, idx):
        return (isinstance(idx, int) and (idx >= 0) and (idx < self.n))

    def moveTensor(self, begIndex, endIndex, warningFlag = True):
        '''
        move then tensor at begIndex to endIndex
        '''
        funcName = 'FreeBoundaryMPS.moveTensor'
        assert (self.isIndex(begIndex) and self.isIndex(endIndex)), funcs.errorMessage("{} or {} is invalid index.".format(begIndex, endIndex), location = funcName)

        if (begIndex == endIndex):
            if (warningFlag):
                warnings.warn(funcs.warningMessage("begIndex and endIndex is equal, do nothing.", location = funcName))
            return 
        
        if (begIndex < endIndex):
            for i in range(begIndex, endIndex):
                self.swap(i, i + 1)
        else:
            for i in range(begIndex, endIndex, -1):
                self.swap(i, i - 1)

    def makeAdjacent(self, idx1, idx2):
        funcName = 'FreeBoundaryMPS.makeAdjacent'
        assert (self.isIndex(idx1) and self.isIndex(idx2)), funcs.errorMessage("{} or {} is invalid index.".format(idx1, idx2), location = funcName)
        assert (idx1 != idx2), funcs.errorMessage("cannot make two identical indices adjacent: {} and {}.".format(idx1, idx2), location = funcName)
        if (idx1 > idx2):
            idx1, idx2 = idx2, idx1 
        for i in range(idx1, idx2 - 1):
            self.swap(i, i + 1)
        return idx2 - 1, idx2
    
    def mergeTensor(self, idx, newTensor):
        # merge idx & (idx + 1) to newTensor
        funcName = 'FreeBoundaryMPS.mergeTensor'
        assert (self.isIndex(idx) and self.isIndex(idx + 1)), funcs.errorMessage("{} or {} is invalid index.".format(idx, idx + 1), location = funcName)
        self._tensors = self._tensors[:idx] + [newTensor] + self._tensors[(idx + 2):]

    def hasTensor(self, tensor):
        return tensor in self._tensors

    def tensorIndex(self, tensor):
        return self._tensors.index(tensor)

    def toTensor(self):
        return contractWithSequence(self._tensors)

def commonLegs(mpsA, mpsB):
    indexA = []
    indexB = []
    # print(mpsA, mpsB)
    for idx in range(mpsA.n):
        tensor = mpsA.getTensor(idx)
        leg = tensor.getLeg('o')
        if (leg.bond is not None) and (mpsB.hasTensor(leg.anotherSide().tensor)):
            idxB = mpsB.tensorIndex(leg.anotherSide().tensor)
            indexA.append(idx)
            indexB.append(idxB)
    return indexA, indexB

def commonBonds(mpsA, mpsB):
    bonds = []
    for tensor in mpsA._tensors:
        leg = tensor.getLeg('o')
        if (leg.bond is not None):
            tensorB = leg.anotherSide().tensor 
            if (mpsB.hasTensor(tensorB)):
                bonds.append(leg.bond)
    
    return bonds

def contractMPS(mpsA, mpsB):
    '''
    solution 0.
    step 0. find all the connections between mpsA and mpsB(in some of o's, the same number)
    step 1. make them continuous on both mps, merge them(merge 2, canonicalize, ...?)
    step 2. use swap to move tensors to one end(tail of mpsA, head of mpsB)
    step 3. merge two tensors, and eliminate the 2-way tensor(to one side)
    Problem: how about canonicalization?
    Only one bond should exist!

    solution 1. we need to save the MPS information in tensors
    extend as MPSTensor
    problem: Schimdt will return a Tensor?
    contractTwoTensors need to maintain MPS information?
    maybe solution: to write a wrapper on these functions maintaining MPS information of Tensor

    solution 2. "merge" for all pairs after one contraction(used here)
    after the contraction, merge the new MPS(mergeMPS) with all existing MPSes
    this may increase the cost of finding merges
    but at the same time, make tensors not need to save MPS information, and for wider usage
    so in this function: no need for considering this task
    '''
    funcName = 'CTL.examples.MPS.contractMPS'
    indexA, indexB = commonLegs(mpsA, mpsB)
    # print('indexA = {}, indexB = {}'.format(indexA, indexB))
    # print('mpsA = {}, mpsB = {}'.format(mpsA, mpsB))
    assert (len(indexA) == 1), funcs.errorMessage("contractMPS can only work on two MPSes sharing one bond, {} obtained.".format((indexA, indexB)), location = funcName)
    if (mpsA.chi != mpsB.chi):
        warnings.warn(funcs.warningMessage(warn = "chi for two MPSes are not equal: {} and {}, choose minimum for new chi.".format(mpsA.chi, mpsB.chi), location = funcName))
    indexA = indexA[0]
    indexB = indexB[0]

    mpsA.moveTensor(indexA, mpsA.n - 1, warningFlag = False)
    mpsB.moveTensor(indexB, 0, warningFlag = False)
    # print('mpsA after swap = {}'.format(mpsA))
    # print('mpsB after swap = {}'.format(mpsB))

    tensorA = mpsA.getTensor(mpsA.n - 1)
    tensorB = mpsB.getTensor(0)

    newTensor = contractTwoTensors(tensorA, tensorB)
    if (newTensor.dim == 0):
        return newTensor
    
    # otherwise, there must be tensors remaining in A or B
    if (mpsA.n > 1):
        newTensor = contractTwoTensors(mpsA.getTensor(-2), newTensor)
        newTensorList = mpsA._tensors[:(-2)] + [newTensor] + mpsB._tensors[1:]
    else:
        newTensor = contractTwoTensors(newTensor, mpsB.getTensor(1))
        newTensorList = mpsA._tensors[:(-1)] + [newTensor] + mpsB._tensors[2:]

    return FreeBoundaryMPS(newTensorList, chi = min(mpsA.chi, mpsB.chi))

def doubleMerge(mpsA, mpsB, idxA, idxB):

    funcName = 'CTL.examples.MPS.doubleMerge'
    tensorA = contractTwoTensors(mpsA.getTensor(idxA), mpsA.getTensor(idxA + 1))
    tensorB = contractTwoTensors(mpsB.getTensor(idxB), mpsB.getTensor(idxB + 1))

    tensorA, tensorB = merge(tensorA, tensorB, chi = None, bondName = 'o')
    mpsA.mergeTensor(idxA, tensorA)
    mpsB.mergeTensor(idxB, tensorB)

    mpsA.canonicalize(idx = idxA)
    mpsB.canonicalize(idx = idxB)

    tensorA, tensorB = mpsA.getTensor(idxA), mpsB.getTensor(idxB)
    tensorA, tensorB = merge(tensorA, tensorB, chi = min(mpsA.chi, mpsB.chi), bondName = 'o', renameWarning = False)
    mpsA.setTensor(idxA, tensorA)
    mpsB.setTensor(idxB, tensorB)
    
    sb = shareBonds(tensorA, tensorB)
    assert (len(sb) == 1), funcs.errorMessage("{} and {} do not share exactly one bond.".format(tensorA, tensorB), location = funcName)
    return sb[0]

def getBondTensorIndices(mpsA, mpsB, bond):
    funcName = 'CTL.examples.MPS.getBondTensorIndices'
    legA, legB = bond.legs 
    tensorA, tensorB = legA.tensor, legB.tensor 
    if (not mpsA.hasTensor(tensorA)):
        tensorA, tensorB = tensorB, tensorA

    assert (mpsA.hasTensor(tensorA)), funcs.errorMessage('{} does not contain {}.'.format(mpsA, tensorA), location = funcName)
    assert (mpsB.hasTensor(tensorB)), funcs.errorMessage('{} does not contain {}.'.format(mpsA, tensorB), location = funcName)

    return mpsA.tensorIndex(tensorA), mpsB.tensorIndex(tensorB)

def doubleMergeByBond(mpsA, mpsB, bond1, bond2):
    funcName = 'CTL.examples.MPS.doubleMergeByBond'
    idxA1, idxB1 = getBondTensorIndices(mpsA, mpsB, bond1)
    idxA2, idxB2 = getBondTensorIndices(mpsA, mpsB, bond2)

    idxA1, idxA2 = mpsA.makeAdjacent(idxA1, idxA2)
    idxB1, idxB2 = mpsB.makeAdjacent(idxB1, idxB2)

    # print('mpsA after swap = {}'.format(mpsA))
    # print('mpsB after swap = {}'.format(mpsB))

    assert (idxA1 + 1 == idxA2) and (idxB1 + 1 == idxB2), funcs.errorMessage("index is not adjacent after swapping: ({}, {}) and ({}, {}).".format(idxA1, idxA2, idxB1, idxB2), location = funcName)
    return doubleMerge(mpsA, mpsB, idxA1, idxB1)

def mergeMPS(mpsA, mpsB, beginFlag = True):
    '''
    merge the common bonds between mpsA and mpsB to one bond
    and merge all the corresponding tensors
    if beginFlag = False, then there should be constraint that the common bonds should not be over 2
    however, this function can handle any number of common bonds
    it traces the bonds(instead of indices) when merging, each time choose two bonds and merge them
    until there are one bond left
    '''
    funcName = 'CTL.examples.MPS.mergeMPS'
    # indexA, indexB = commonLegs(mpsA, mpsB)
    bonds = commonBonds(mpsA, mpsB)
    if (len(bonds) <= 1):
        # only 0 or 1 common bonds, no need for merge
        return 
    assert (beginFlag or (len(bonds) == 2)), funcs.errorMessage(err = "there should be no more than 2 common bonds between MPSes {} and {} if not at beginning: {} obtained.".format(mpsA, mpsB, bonds), location = funcName)

    # what we want to do here: merge indices in indexA and indexB
    # if only two indices: we can do it as following

    # bonds = []
    # for idxA, idxB in zip(indexA, indexB):
    #     tensorA = mpsA.getTensor(idxA)
    #     tensorB = mpsB.getTensor(idxB)
    #     sb = shareBonds(tensorA, tensorB)
    #     assert (len(sb) == 1), funcs.errorMessage("{} and {} do not share bonds.".format(tensorA, tensorB), location = funcName)
    #     bonds.append(sb[0])

    while (len(bonds) > 1):
        bond1, bond2 = bonds[0], bonds[1]
        newBond = doubleMergeByBond(mpsA, mpsB, bond1, bond2)
        bonds = [newBond] + bonds[2:]

    # otherwise, we need to repeat doubleMerge until there is one left
    # use the bond to find the tensors?

def createMPSFromTensor(tensor, chi = 16):
    '''
    tensor is a real Tensor with n outer legs
    transfer it into an MPS with Schimdt decomposition
    after this, we can manage the tensor network decomposition by MPS network decomposition
    finally consider if tensor is only a TensorLike object
    '''

    # TODO: make this function work for tensorLike

    funcName = 'CTL.examples.MPS.createMPSFromTensor'

    legs = [leg for leg in tensor.legs]
    # xp = tensor.xp 

    n = len(legs)
    assert (n > 0), funcs.errorMessage("cannot create MPS from 0-D tensor {}.".format(tensor), location = funcName)

    if (n == 1):
        warnings.warn(funcs.warningMessage("creating MPS for 1-D tensor {}.".format(tensor), location = funcName), RuntimeWarning)
        return FreeBoundaryMPS([tensor], chi = chi)

    a = xplib.xp.ravel(tensor.toTensor(labels = None))
    
    lastDim = -1
    tensors = []
    lastRightLeg = None
    for i in range(n - 1):
        u, v = matrixSchimdtDecomposition(a, dim = legs[i].dim, chi = chi)
        leg = legs[i]
        if (i == 0):
            dim1 = u.shape[1]
            rightLeg = Leg(None, dim = dim1, name = 'r')
            tensor = Tensor(shape = (leg.dim, u.shape[1]), legs = [leg, rightLeg], data = u)
            lastRightLeg = rightLeg
            lastDim = dim1
        else:
            dim1 = u.shape[-1]
            leftLeg = Leg(None, dim = lastDim, name = 'l')
            rightLeg = Leg(None, dim = dim1, name = 'r')
            tensor = Tensor(shape = (lastDim, leg.dim, u.shape[-1]), legs = [leftLeg, leg, rightLeg], data = u)

            makeLink(leftLeg, lastRightLeg)

            lastRightLeg = rightLeg 
            lastDim = dim1
        
        tensors.append(tensor)
        a = v

    leftLeg = Leg(None, dim = lastDim, name = 'l')
    tensor = Tensor(shape = (lastDim, legs[-1].dim), legs = [leftLeg, legs[-1]], data = a)
    makeLink(leftLeg, lastRightLeg)
    tensors.append(tensor)

    # print(tensors)

    return FreeBoundaryMPS(tensorList = tensors, chi = chi)

def contractWithMPS(tensorList, chi = 16, seq = None, greedyFlag = True):
    # DONE: change optimal sequence to greedy sequence
    if (seq is None):
        if (greedyFlag):
            seq = generateGreedySequence(tensorList)
            # print('greedy sequence = {}'.format(seq))
        else:
            seq = generateOptimalSequence(tensorList)
    # print('sequence = {}'.format(seq))
    mpses = [createMPSFromTensor(tensor, chi = chi) for tensor in tensorList]

    n = len(tensorList)
    for i in range(n):
        for j in range(i + 1, n):
            mergeMPS(mpses[i], mpses[j], beginFlag = True)

    for s, t in seq:
        # print('mpses = {}'.format(mpses))
        # print('contracting ({}, {})'.format(s, t))
        # if (len(mpses) > 1) and (mpses[0] is not None) and (mpses[1] is not None):
        #     indexA, indexB = commonLegs(mpses[0], mpses[1])
        #     print('common legs: {}, {}'.format(indexA, indexB))
        loc = min(s, t)
        mpses[loc] = contractMPS(mpses[s], mpses[t])
        mpses[s + t - loc] = None 
        for i in range(n):
            if (i != loc) and (mpses[i] is not None):
                mergeMPS(mpses[loc], mpses[i], beginFlag = False)

        # indexA, indexB = commonLegs(mpses[0], mpses[1])

    if (isinstance(mpses[0], FreeBoundaryMPS)):
        return mpses[0].toTensor()
    else:
        return mpses[0]