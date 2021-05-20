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

        # if real cost, then we will always calculate the optimal sequence with real bond dims
        # then for each time the bond dimension changes, we need to recalculate the sequence
        if (realCost):
            self.typicalDim = None 
        else:
            self.typicalDim = typicalDim

    @property 
    def tensorCount(self):
        return len(self.tensorNames)
    
    def addTensor(self, name):
        assert (not self.locked), "Error: adding tensor {} to a FTN which is locked.".format(name)
        self.tensorNames.append(name)
    def removeTensor(self, name):
        assert (not self.locked), "Error: removing tensor {} to a FTN which is locked.".format(name)
        assert (name in self.tensorNames), "Error: removing tensor {} to a FTN which do not contain it.".format(name)
        self.tensorNames.remove(name)

    def lock(self):
        self.locked = True 
    def unlock(self):
        self.locked = False

    def addLink(self, aName, aLeg, bName, bLeg):
        assert (not self.locked), "Error: adding link ({}, {}) to a FTN which is locked.".format((aName, aLeg), (bName, bLeg))
        self.links.append(((aName, aLeg), (bName, bLeg)))

    def loadBondDims(self, localTensors):
        for name in localTensors:
            for leg in localTensors[name].legs:
                legName = name + '-' + leg.name 
                if (legName not in self.bondDims) or (leg.dim != self.bondDims[legName]):
                    self.changed = True
                self.bondDims[legName] = leg.dim

    def contract(self, tensorDict):
        self.lock()

        assert funcs.compareLists(self.tensorNames, list(tensorDict.tensors.keys())), "Error: input tensorDict {} does not compatible with FTN {}.".format(list(tensorDict.tensors.keys()), self.tensorNames)

        localTensors = dict()
        for name in tensorDict.tensors:
            localTensors[name] = tensorDict.tensors[name].copy()

        for leg1, leg2 in self.links:
            tensorA, legA = leg1 
            tensorB, legB = leg2 
            makeLink(legA, legB, localTensors[tensorA], localTensors[tensorB])

        self.changed = False 
        self.loadBondDims(localTensors)

        tensorList = [localTensors[name] for name in self.tensorNames]
        if (self.optimalSeq is None) or (self.realCost and self.changed):
            self.optimalSeq = generateOptimalSequence(tensorList, bf = False, typicalDim = self.typicalDim)

        res = contractWithSequence(tensorList, self.optimalSeq)
        return res
        

        

    


    
    
