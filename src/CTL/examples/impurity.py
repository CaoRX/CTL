# TODO: impurity tensor calculation
# input a set of impurity tensors(order 0, 1, 2...) to a framework like HOTRG
# calculate the impurity average up to any order

import CTL.funcs.funcs as funcs # divideIntoKParts(n, k), calculateDivisionTimes needed
from CTL.tensor.tensor import Tensor

class ImpurityTensorNetwork:

    def __init__(self, impurities, highestOrder = None):
        self.impurities = [tensor.copy() for tensor in impurities]
        if (highestOrder is None):
            highestOrder = len(impurities)
        self.highestOrder = highestOrder # 0 .. order - 1
        if (len(self.impurities) > self.highestOrder):
            self.impurities = self.impurities[:self.highestOrder]
        else:
            self.impurities += [None] * (self.highestOrder - len(self.impurities))

        self.RG = None 
        self.impurityArchive = []
        self.norms = []
        self.iterateIdx = 0

        self.initialized = False
    
    def setRG(self, rg):
        self.RG = rg 
        # RG must have: iterate, impurityIterate
        # self.RG.impurityIterate(tensors, loopIdx)

    def appendToArchive(self, norm):
        # print('norm = {}'.format(norm))
        for a in self.impurities:
            a.a /= norm 
        self.norms.append(norm)
        self.impurityArchive.append([a.copy() for a in self.impurities])

    def initialize(self):
        assert (self.RG is not None), "Error: ImpurityTensorNetwork cannot initialize if RG is None."
        # here we only consider normalization with original norms
        # further implementation will be done next
        self.appendToArchive(self.RG.getNorm(0))
        self.initialized = True

    def iterate(self):
        if (not self.initialized):
            self.initialize()
        # iterate impurity to a new set of impurity tensors
        newImpurities = []
        norm = self.RG.getNorm(self.iterateIdx + 1)
        partN = self.RG.impurityParts
        newDOF = partN * self.impurities[0].degreeOfFreedom
        for i in range(self.highestOrder):
            weight = 0.5 ** i 
            totalWeight = 0.0
            resA = None
            resLabels = None
            for parts in funcs.divideIntoKParts(i, partN):
                existFlag = True
                for j in parts:
                    if (self.impurities[j] is None):
                        existFlag = False 
                if (not existFlag):
                    continue
                partWeight = funcs.calculateDivisionTimes(parts) * weight 
                totalWeight += partWeight 

                tensors = []
                for j in parts:
                    tensors.append(self.impurities[j])
                res = self.RG.impurityIterate(tensors, self.iterateIdx)

                if (resLabels is None):
                    resLabels = res.labels 
                    resA = res.a * partWeight 
                else:
                    res.reArrange(resLabels)
                    resA += res.a * partWeight 
            
            resA /= totalWeight 
            resTensor = Tensor(labels = resLabels, data = resA, degreeOfFreedom = newDOF)
            newImpurities.append(resTensor)

        self.iterateIdx += 1
        self.impurities = newImpurities
        self.appendToArchive(norm)
    
    def measureObservables(self):
        res = []
        for i in range(len(self.impurityArchive)):
            res.append([])
            for tensor in self.impurityArchive[i]:
                res[-1].append(self.RG.tensorTrace(tensor) / self.RG.pureTensorTrace(i))

        return res


