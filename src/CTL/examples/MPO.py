# Matrix Product Operators
# The shape is similar to MPS
# start with a rank-3 tensor, and then by rank-4 tensors until the end with rank-3 tensor

from venv import create
import CTL.funcs.xplib as xplib
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.contract import contractTwoTensorsNotInPlace, shareBonds, contractTwoTensors, merge
from CTL.tensor.tensor import Tensor
from CTL.tensor.leg import Leg
from CTL.tensor.contract.optimalContract import copyTensorList, contractWithSequence, generateOptimalSequence, generateGreedySequence
from CTL.examples.Schimdt import SchimdtDecomposition, matrixSchimdtDecomposition
from CTL.tensor.tensorFunc import isIsometry
import warnings
from CTL.tensor.contract.link import makeLink
from CTL.examples.MPS import FreeBoundaryMPS, createApproxMPS
from CTL.tensor.contract.contractExp import MPOApplyFTN, MPOApplyLeftFTN, MPOApplyRightFTN, MPOInnerProductFTN, MPOInnerProductLeftSideFTN, MPOInnerProductRightSideFTN, MPOInnerProductOnly1FTN, MPOHorizontalFTN

class FreeBoundaryMPO:

    '''
    MPO:
    Maintain a list of tensors, the first & last tensors are 3-dimensional, others are 4-dimensional
    The outer bonds are just the legs of the input tensor, and the internal bonds are with bond dimension at max chi
    
    Each tensor has an "u" and "d" for top and bottom leg, so it can be inserted with an MPS to make a new MPS.
    '''

    def checkMPOProperty(self, tensorList):
        # MPO property: first and last tensor has two bonds out, and another linked to the next
        # others: one to left, one to right, two out

        n = len(tensorList)

        if (n == 0):
            return False 

        for i in range(n - 1):
            if (len(shareBonds(tensorList[i], tensorList[i + 1])) != 1):
                return False 
        
        if (n > 1) and ((tensorList[0].dim != 3) or (tensorList[-1].dim != 3)):
            return False
        
        for i in range(1, n - 1):
            if (tensorList[i].dim != 4):
                return False 

        for tensor in tensorList:
            if not funcs.compareLists(tensor.getFreeLabels(), ['u', 'd']):
                return False
        
        return True

    def getChi(self, chi):
        # if chi is None: then take the maximum from bonds shared
        # otherwise, if bonds sharing larger than chi, then warning and update chi
        # otherwise, take chi
        bondChi = -1
        for i in range(self.n - 1):
            bond = shareBonds(self._tensors[i], self._tensors[i + 1])[0]
            if bondChi == -1:
                bondChi = bond.legs[0].dim
            else:
                bondChi = max(bondChi, bond.legs[0].dim) 
        
        if (chi is None):
            return bondChi 
        elif (bondChi > chi):
            warnings.warn(funcs.warningMessage('required chi {} is lower than real bond chi {}, set to {}.'.format(chi, bondChi, bondChi), location = "FreeBoundaryMPO.getChi"))
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

        # for i in range(self.n):
        #     for leg in self._tensors[i].legs:
        #         if (leg.bond not in self.internalBonds):
        #             leg.name = 'o'

    def __init__(self, tensorList, chi = None, inplace = True):
        if (not self.checkMPOProperty(tensorList)):
            raise ValueError(funcs.errorMessage("tensorList {} cannot be considered as an MPO".format(tensorList), location = "FreeBoundaryMPO.__init__"))
        
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

        self.innerProductFTN = MPOInnerProductFTN()
        self.ipLeftFTN = MPOInnerProductLeftSideFTN()
        self.ipRightFTN = MPOInnerProductRightSideFTN()
        self.ipSingleFTN = MPOInnerProductOnly1FTN()

        self.l2rFTN = MPOHorizontalFTN()
        self.l2rEndFTN = MPOHorizontalFTN()
        self.r2lFTN = MPOHorizontalFTN()
        self.r2lEndFTN = MPOHorizontalFTN()

        self.applyFTN = {
            'u': MPOApplyFTN('u'), 
            'd': MPOApplyFTN('d')
        }

        self.applyLeftFTN = {
            'u': MPOApplyLeftFTN('u'), 
            'd': MPOApplyLeftFTN('d')
        }

        self.applyRightFTN = {
            'u': MPOApplyRightFTN('u'), 
            'd': MPOApplyRightFTN('d')
        }

    
    @property 
    def n(self):
        return len(self._tensors)
    def __repr__(self):
        return 'FreeBoundaryMPO(tensors = {}, chi = {})'.format(self._tensors, self.chi)

    def setTensor(self, idx, tensor):
        assert (idx >= 0 and idx < self.n), funcs.errorMessage("index must be in [0, n) but {} obtained.".format(idx), location = "FreeBoundaryMPO.setTensor")
        self._tensors[idx] = tensor 
        self.activeIdx = None
    def getTensor(self, idx):
        return self._tensors[idx]
    def isIndex(self, idx):
        return (isinstance(idx, int) and (idx >= 0) and (idx < self.n))

    def hasTensor(self, tensor):
        return tensor in self._tensors

    def tensorIndex(self, tensor):
        return self._tensors.index(tensor)

    def toTensor(self):
        return contractWithSequence(self._tensors)

    def copyOfTensors(self):
        tensors = [self.getTensor(i).copy() for i in range(self.n)]
        for i in range(self.n - 1):
            makeLink('r', 'l', tensors[i], tensors[i + 1])
        return tensors

    def checkMPSCompatible(self, mps, direction = 'u'):
        '''
        check whether the input mps has a compatible shape with current MPO
        '''

        location = 'FreeBoundaryMPO.checkMPSCompatible'

        if direction not in ['u', 'd']:
            raise ValueError(funcs.errorMessage(
                'MPO can only be applied to MPS in direction either "u" or "d", but {} obtained'.format(direction),
                location = location
            ))
        
        if mps.n != self.n:
            return False

        for mpsTensor, mpoTensor in zip(mps._tensors, self._tensors):
            if mpsTensor.shapeOfLabel('o') != mpoTensor.shapeOfLabel(direction):
                return False

        return True

    def oppositeDirection(self, d):
        if d == 'u':
            return 'd'
        elif d == 'd':
            return 'u'
        else:
            raise ValueError(funcs.errorMessage('MPO can only flip directions between "d" and "u", {} obtained'.format(d), location = 'FreeBoundaryMPO.oppositeDirection'))

    def applyToMPS(self, mps, direction = 'u', newChi = None):
        # print('apply {} to {}'.format(self, mps))
        if not self.checkMPSCompatible(mps, direction = direction):
            raise ValueError(funcs.errorMessage("Input MPS {} and MPO {} is not compatible, None returned".format(mps, self), location = 'FreeBoundaryMPO.applyToMPS'))

        resTensorList = []


        for i, mpsTensor, mpoTensor in zip(range(self.n), mps._tensors, self._tensors):
            # bonds = makeLink('o', direction, mpsTensor, mpoTensor)
            # resTensor = contractTwoTensorsNotInPlace(mpsTensor, mpoTensor)
            # resTensor.renameLabel(self.oppositeDirection(direction), 'o')
            if i == 0:
                ftn = self.applyLeftFTN[direction]
            elif i == self.n - 1:
                ftn = self.applyRightFTN[direction]
            else:
                ftn = self.applyFTN[direction]
            resTensor = ftn.contract({
                'mpo': mpoTensor, 
                'mps': mpsTensor
            })
            resTensorList.append(resTensor)
            # for bond in bonds:
            #     bond.clear()

        resN = self.n
        for i in range(resN - 1):
            makeLink('r', 'l', resTensorList[i], resTensorList[i + 1])
        
        if newChi is None:
            newChi = self.chi * mps.chi

        return createApproxMPS(tensorList = resTensorList, chi = newChi, inplace = True)
        # return FreeBoundaryMPS(tensorList = resTensorList, chi = newChi, inplace = True)

    def innerProduct(self, mpsU, mpsD, remainIndex = None):
        # contract with two MPSes
        # if remainIndex is None: return a single tensor(scalar)
        # otherwise, The tensors will be contracted without mpsU/D._tensors[remainIndex]
        # so the result will be a dim-6(or dim 4 if at the ends) tensor
        # this will need a speed-up, since we can save partial results
        # but here we only consider full contraction

        location = 'FreeBoundaryMPO.innerProduct'
        if not self.checkMPSCompatible(mpsU, 'u'):
            raise ValueError(funcs.errorMessage("upper MPS {} is not compatible with current MPO {}.".format(mpsU, self), location = location))
        if not self.checkMPSCompatible(mpsD, 'd'):
            raise ValueError(funcs.errorMessage("lower MPS {} is not compatible with current MPO {}.".format(mpsD, self), location = location))

        if remainIndex is None:
            resTensors = []
            for i in range(self.n):
                upTensor = mpsU._tensors[i]
                downTensor = mpsD._tensors[i]
                mpoTensor = self._tensors[i]

                ftn = self.innerProductFTN
                if self.n == 1:
                    ftn = self.ipSingleFTN
                elif i == 0:
                    ftn = self.ipLeftFTN
                elif i == self.n - 1:
                    ftn = self.ipRightFTN

                resTensor = ftn.contract({
                    'u': upTensor, 
                    'd': downTensor, 
                    'mpo': mpoTensor
                })
                resTensors.append(resTensor)
            
            res = resTensors[0]
            for i in range(1, self.n - 1):
                res = self.l2rFTN.contract({
                    'l': res, 
                    'r': resTensors[i]
                })
            if self.n > 1:
                res = self.l2rEndFTN.contract({
                    'l': res, 
                    'r': resTensors[self.n - 1]
                })
            
            return res

        # otherwise, we should keep the tensors on remainIndex
        # but this will actually cost more time
        # this will not be implemented, and more efficient method should be used for this task
        raise ValueError(funcs.errorMessage("remainIndex is not supported, since it will bring worse efficiency compared with suffix-sum methods", location = location))

        # TODO: add tests for MPO!
                
def identityMPO(n, dim = 2):
    location = 'CTL.bin.HeisenbergMPO.identityMPO'
    if (n < 2):
        raise ValueError(funcs.errorMessage("identity MPO must have length > 1, got {}".format(n), location = location))
    
    tensors = []
    for i in range(n):

        if i == 0:
            shape = (dim, dim, 1)
            labels = ['u', 'd', 'r']
        elif i == (n - 1):
            # data = np.eye(n).reshape((dim, dim, 1))
            shape = (dim, dim, 1)
            labels = ['u', 'd', 'l']
        else:
            shape = (dim, dim, 1, 1)
            labels = ['u', 'd', 'l', 'r']

        data = xplib.xp.eye(dim).reshape(shape)
        tensor = Tensor(shape = shape, labels = labels, data = data)
        tensors.append(tensor)

    for i in range(n - 1):
        makeLink('r', 'l', tensors[i], tensors[i + 1])
    
    return FreeBoundaryMPO(tensorList = tensors)
        
