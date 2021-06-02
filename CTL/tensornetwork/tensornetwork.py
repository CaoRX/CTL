from CTL.tensor.contract.link import makeLink
import CTL.funcs.funcs as funcs
from CTL.tensor.contract.optimalContract import generateOptimalSequence, contractWithSequence
class FiniteTensorNetwork:
    
    # instead of save the tensors: save a set of operations
    # provide a function api FiniteTensorNetwork.contract(tensorDict)

    def __init__(self, tensorNames = [], realCost = True, typicalDim = 10):
        self.tensorNames = tensorNames
        self.optimalSeq = None 
        self.realCost = realCost 
        self.links = []

        self.bondDims = dict([])
        self.locked = False
        self.changed = False

        self.changeNameBefore = []
        self.changeNameAfter = []
        self.outProductAfter = []

        # if real cost, then we will always calculate the optimal sequence with real bond dims
        # then for each time the bond dimension changes, we need to recalculate the sequence
        if (realCost):
            self.typicalDim = None 
        else:
            self.typicalDim = typicalDim

    @property 
    def tensorCount(self):
        return len(self.tensorNames)

    def checkLock(self, opt):
        assert (not self.locked), "Error: {} to a FTN which is locked.".format(opt)
    
    def addTensor(self, name):
        # assert (not self.locked), "Error: adding tensor {} to a FTN which is locked.".format(name)
        self.checkLock('adding tensor {}'.format(name))
        self.tensorNames.append(name)
    def removeTensor(self, name):
        # assert (not self.locked), "Error: removing tensor {} to a FTN which is locked.".format(name)
        self.checkLock('removing tensor {}'.format(name))
        assert (name in self.tensorNames), "Error: removing tensor {} to a FTN which do not contain it.".format(name)
        self.tensorNames.remove(name)

    def lock(self):
        self.locked = True 
    def unlock(self):
        self.locked = False

    def addLink(self, aName, aLeg, bName, bLeg):
        # assert (not self.locked), "Error: adding link ({}, {}) to a FTN which is locked.".format((aName, aLeg), (bName, bLeg))
        self.checkLock('adding link ({}, {})'.format((aName, aLeg), (bName, bLeg)))
        self.links.append(((aName, aName + '-' + aLeg), (bName, bName + '-' + bLeg)))

    def addPreNameChange(self, name, legName, newName):
        self.checkLock('add pre name change for tensor {}: from {} to {}'.format(name, legName, newName))
        self.changeNameBefore.append((name, legName, newName))
    def addPostNameChange(self, name, legName, newName):
        self.checkLock('add post name change for tensor {}: from {} to {}'.format(name, legName, newName))
        self.changeNameAfter.append((name, legName, newName))

    def removePreNameChange(self, name, legName, newName):
        self.checkLock('remove pre name change for tensor {}: from {} to {}'.format(name, legName, newName))
        self.changeNameBefore.remove((name, legName, newName))
    def removePostNameChange(self, name, legName, newName):
        self.checkLock('remove post name change for tensor {}: from {} to {}'.format(name, legName, newName))
        self.changeNameAfter.remove((name, legName, newName))

    def addPostOutProduct(self, labelList, newLabel):
        self.checkLock('add post out product: {} -> {}'.format(labelList, newLabel))
        self.outProductAfter.append((labelList, newLabel))

    def loadBondDims(self, localTensors):
        for name in localTensors:
            for leg in localTensors[name].legs:
                # legName = name + '-' + leg.name 
                if (leg.name not in self.bondDims) or (leg.dim != self.bondDims[leg.name]):
                    self.changed = True
                self.bondDims[leg.name] = leg.dim

    def _dealOutProductLabel(self, label):
        if (isinstance(label, tuple)):
            return label[0] + '-' + label[1]
        else:
            return label

    def contract(self, tensorDict, removeTensorTag = True):
        self.lock()

        assert funcs.compareLists(self.tensorNames, list(tensorDict.tensors.keys())), "Error: input tensorDict {} does not compatible with FTN {}.".format(list(tensorDict.tensors.keys()), self.tensorNames)

        localTensors = dict()
        for name in tensorDict.tensors:
            localTensors[name] = tensorDict.tensors[name].copy()

        for tensor, legName, newName in self.changeNameBefore:
            localTensors[tensor].renameLabel(legName, newName)

        for name in localTensors:
            localTensors[name].addTensorTag(name)

        for leg1, leg2 in self.links:
            tensorA, legA = leg1 
            tensorB, legB = leg2 
            makeLink(legA, legB, localTensors[tensorA], localTensors[tensorB])

        self.changed = False 
        self.loadBondDims(localTensors)

        tensorList = [localTensors[name] for name in self.tensorNames]
        if (self.optimalSeq is None) or (self.realCost and self.changed):
            self.optimalSeq = generateOptimalSequence(tensorList, bf = False, typicalDim = self.typicalDim)

        res = contractWithSequence(tensorList, seq = self.optimalSeq, inplace = True)
        # print(res)
        # print(tensorDict.tensors)

        for labelList, newLabel in self.outProductAfter:
            labelList = [self._dealOutProductLabel(label) for label in labelList]
            # print(labelList, newLabel)
            # print(res.labels)
            res.outProduct(labelList, 'res-' + newLabel)
            # print(res.labels)

        for tensor, legName, newName in self.changeNameAfter:
            res.renameLabel(tensor + '-' + legName, tensor + '-' + newName)
        
        if (removeTensorTag):
            res.removeTensorTag()
        return res
        

        

    


    
    
