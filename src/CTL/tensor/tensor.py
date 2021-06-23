from CTL.tensorbase.tensorbase import TensorBase 
import CTL.funcs.funcs as funcs
import numpy as np 
from copy import deepcopy
from CTL.tensor.leg import Leg

class Tensor(TensorBase):

    # def __init__(self):
    #     pass
    # attributes: xp, totalSize, labels, contractLabels, degreeOfFreedom, a

    # bondNameSet = set([])

    def __init__(self, shape = None, labels = None, data = None, degreeOfFreedom = None, name = None, legs = None):
        super().__init__(None)
        # print('Tensor(shape = {}, labels = {}, data.shape = {})'.format(shape, labels, data.shape))
        assert (not ((data is None) and (shape is None))), "Error: TensorBase must be initialized with either data or shape."
        if (shape is None):
            shape = data.shape
        # print(shape, data.shape)

        assert ((labels is None) or (len(shape) == len(labels))), "Error: the number of labels input is {}, while the dimension is {}.".format(len(labels), len(shape))

        self.xp = np
        # self.shape = shape
        self.totalSize = funcs.tupleProduct(shape)
        if (data is None):
            #self.a = self.xp.zeros(self.shape, dtype = self.xp.float64)
            self.a = self.xp.random.random_sample(shape)
        else:
            self.a = self.xp.copy(data)
        assert (self.totalSize == funcs.tupleProduct(self.a.shape)), 'Error: expect {} elements but {} gotten.'.format(self.totalSize, funcs.tupleProduct(self.a.shape))
        if (self.a.shape != shape):
            self.a = self.xp.reshape(self.a, shape)
        # print('a shape = {}'.format(self.a.shape))
    
        if (labels is None):
            labels = self.generateLabels(len(shape))
        self.degreeOfFreedom = degreeOfFreedom
        self.name = name

        if (legs is None):
            self.legs = []
            for label, dim in zip(labels, list(self.shape)):
                self.legs.append(Leg(self, dim, label))
        else:
            assert (len(legs) == self.dim), "Error: number of legs and dim are not compatible in Tensor.__init__(): {} and {}.".format(len(legs), self.dim)
            self.legs = legs 
            for leg in self.legs:
                leg.tensor = self

    @property 
    def labels(self):
        return [leg.name for leg in self.legs]
    
    @property 
    def chi(self):
        return self.shape[0]


    def __str__(self):
        if not (self.degreeOfFreedom is None):
            dofStr = ', degree of freedom = {}'.format(self.degreeOfFreedom)
        else:
            dofStr = ''
        if (self.name is not None):
            nameStr = self.name + ', ' 
        else:
            nameStr = ''
        return 'Tensor({}shape = {}, labels = {}{})'.format(nameStr, self.shape, self.labels, dofStr)

    def __repr__(self):
        if not (self.degreeOfFreedom is None):
            dofStr = ', degree of freedom = {}'.format(self.degreeOfFreedom)
        else:
            dofStr = ''
        if (self.name is not None):
            nameStr = self.name + ', ' 
        else:
            nameStr = ''
        return 'Tensor({}shape = {}, labels = {}{})\n'.format(nameStr, self.shape, self.labels, dofStr)

    def bondDimension(self):
        return self.shape[0]

    def generateLabels(self, n):
        assert (n <= 26), "Too many dimensions for input shape"
        labelList = 'abcdefghijklmnopqrstuvwxyz'
        return [labelList[i] for i in range(n)]
    
    def indexOfLabel(self, lab, backward = False):
        labels = self.labels
        if not (lab in labels):
            return -1
        if (backward): # search from backward, for a list can have more than one same label
            labels.reverse()
            ret = len(labels) - labels.index(lab) - 1
            labels.reverse()
        else:
            ret = labels.index(lab)
        return ret

    def getLegIndex(self, leg):
        return self.legs.index(leg)
    def getLegIndices(self, legs):
        return [self.getLegIndex(leg) for leg in legs]
    def getLeg(self, label):
        res = None 
        for leg in self.legs:
            if (leg.name == label):
                res = leg 
                break 
        assert (res is not None), "Error: {} not in tensor labels {}.".format(label, self.labels)
        
        return res

    def toVector(self):
        return self.xp.copy(self.xp.ravel(self.a))
    
    def toMatrix(self, rows, cols):
        # print(rows, cols)
        # print(self.labels)
        # input two set of legs
        assert not ((rows is None) and (cols is None)), "Error in Tensor.toMatrix: toMatrix must have at least row or col exist."
        if (rows is not None) and (isinstance(rows[0], str)):
            rows = [self.getLeg(label) for label in rows]
        if (cols is not None) and (isinstance(cols[0], str)):
            cols = [self.getLeg(label) for label in cols]
        if (cols is None):
            cols = funcs.listDifference(self.legs, rows)
        if (rows is None):
            rows = funcs.listDifference(self.legs, cols)
        assert (funcs.compareLists(rows + cols, self.legs)), "Error Tensor.toMatrix: rows + cols must contain(and only contain) all legs of tensor."

        colIndices = self.getLegIndices(cols)
        rowIndices = self.getLegIndices(rows)

        colShape = tuple([self.shape[x] for x in colIndices])
        rowShape = tuple([self.shape[x] for x in rowIndices])
        colTotalSize = funcs.tupleProduct(colShape)
        rowTotalSize = funcs.tupleProduct(rowShape)

        moveFrom = rowIndices + colIndices
        moveTo = list(range(len(moveFrom)))

        data = self.xp.moveaxis(self.xp.copy(self.a), moveFrom, moveTo)
        data = self.xp.reshape(data, (rowTotalSize, colTotalSize))
        return data

    def complementLegs(self, legs):
        return funcs.listDifference(self.legs, legs)

    def copy(self):
        return Tensor(data = self.xp.copy(self.a), degreeOfFreedom = self.degreeOfFreedom, name = self.name, labels = self.labels)
        # no copy of tensor legs, which may contain connection information
    
    def copyN(self, n):
        return [self.copy() for _ in range(n)]
    
    def getLabelIndices(self, labs, backward = False):
        ret = []
        for i, lab in enumerate(labs):
            if (labs[:i].count(lab) > 0):
                ret.append(self.indexOfLabel(lab, backward = (not backward)))
                # if there are two same labels in labs
                # then the first take from front, the second from back
            else:
                ret.append(self.indexOfLabel(lab, backward = backward))

        return ret

    def renameLabel(self, changeFrom, changeTo):
        self.legs[self.indexOfLabel(changeFrom)].name = changeTo
        # self.legs[changeTo] = self.legs[changeFrom]
        # if (changeFrom != changeTo):
        #     del self.legs[changeFrom]

    def renameLabels(self, changefrom, changeto):
        assert (len(changefrom) == len(changeto)), "Error: renameLabels need two list with equal number of labels, gotten {} and {}".format(changefrom, changeto)
        for cf, ct in zip(changefrom, changeto):
            self.renameLabel(cf, ct)

    def shapeOfLabel(self, label):
        for leg in self.legs:
            if leg.name == label:
                return leg.dim 
        
        return -1
    def shapeOfLabels(self, labs):
        return self.shapeOfIndices(self.getLabelIndices(labs))

    def shapeOfIndex(self, index):
        return self.shape[index]
    def shapeOfIndices(self, indices):
        return tuple([self.shape[x] for x in indices])

    def addTensorTag(self, name):
        for leg in self.legs:
            assert (leg.name.find('-') == -1), "Error: leg name {} already has a tensor tag.".format(leg.name)
            leg.name = name + '-' + leg.name 
    
    def removeTensorTag(self):
        for leg in self.legs:
            divLoc = leg.name.find('-')
            assert (divLoc != -1), "Error: leg name {} does not contain a tensor tag.".format(leg.name)
            leg.name = leg.name[(divLoc + 1):]

    def moveLabelsToFront(self, labelList):
        moveFrom = []
        moveTo = []
        currIdx = 0
        movedLegs = []
        for label in labelList:
            for i, leg in enumerate(self.legs):
                if (leg.name == label):
                    moveFrom.append(i)
                    moveTo.append(currIdx)
                    currIdx += 1
                    movedLegs.append(leg)
                    break

        for leg in movedLegs:
            self.legs.remove(leg)
        
        # print(moveFrom, moveTo)
        # print(labelList)
        # print(self.labels)
        self.legs = movedLegs + self.legs 
        self.a = self.xp.moveaxis(self.a, moveFrom, moveTo)

    def outProduct(self, labelList, newLabel):
        self.moveLabelsToFront(labelList)
        n = len(labelList)
        newShape = (-1, ) + self.shape[n:]
        self.a = np.reshape(self.a, newShape)
        # self.shape = self.a.shape

        newDim = self.a.shape[0]
        self.legs = [Leg(self, newDim, newLabel)] + self.legs[n:]

    def reArrange(self, labels):
        assert (funcs.compareLists(self.labels, labels)), "Error: tensor labels must be the same with original labels: get {} but {} needed".format(len(labels), len(self.labels))
        self.moveLabelsToFront(labels)

    def norm(self):
        return self.xp.linalg.norm(self.a)

    def trace(self, rows = None, cols = None):
        mat = self.toMatrix(rows = rows, cols = cols)
        assert (mat.shape[0] == mat.shape[1]), "Error: Tensor.trace must have the same dimension for cols and rows, but shape {} gotten.".format(mat.shape)
        return self.xp.trace(mat)

    def single(self):
        # return the single value of this tensor
        # only works if shape == (,)
        assert self.shape == (), "Error: cannot get single value from tensor whose shape is not ()."
        return self.a

    def toTensor(self, labels):
        self.reArrange(labels)
        return self.a

    # def complementIndices(self, labs):
    #     return funcs.listDifference(self.labels, labs)

    # def shapeOfRowColumn(self, rows, cols = None):
    #     if (cols == None):
    #         cols = funcs.listDifference(self.labels, rows)

    #     colIndices = self.getLabelIndices(cols)
    #     rowIndices = self.getLabelIndices(rows)

    #     colShape = tuple([self.shape[x] for x in colIndices])
    #     rowShape = tuple([self.shape[x] for x in rowIndices])
    #     return rowShape, colShape

    # def toMatrix(self, rows, cols = None):
    #     if (cols == None):
    #         cols = funcs.listDifference(self.labels, rows)

    #     colIndices = self.getLabelIndices(cols)
    #     rowIndices = self.getLabelIndices(rows, backward = True)

    #     colShape = tuple([self.shape[x] for x in colIndices])
    #     rowShape = tuple([self.shape[x] for x in rowIndices])

    #     colTotalSize = funcs.tupleProduct(colShape)
    #     rowTotalSize = funcs.tupleProduct(rowShape)

    #     moveFrom = rowIndices + colIndices
    #     moveTo = list(range(len(moveFrom)))
    #     #print('row label = {}, col label = {}'.format(rows, cols))
    #     #print('row = {}, col = {}'.format(row_index, col_index))
    #     #print('move from {}, to {}'.format(moveFrom, moveTo))
    #     data = self.xp.moveaxis(self.xp.copy(self.a),  moveFrom, moveTo)
    #     data = self.xp.reshape(data, (rowTotalSize, colTotalSize))
    #     return data

    # def toTensor(self, labels):
    #     assert (len(labels) == len(self.labels)), "Error: number of tensor labels must be the same with original labels: get {} but {} needed".format(len(labels), len(self.labels))

    #     moveFrom = self.getLabelIndices(labels)
    #     moveTo = list(range(len(moveFrom)))
    #     data = self.xp.moveaxis(self.xp.copy(self.a),  moveFrom, moveTo)
    #     return data

    # # def moveContractLabels(self, moveFrom, moveTo):
    # #     contractLabels = [''] * self.dim 
    # #     for mf, mt in zip(moveFrom, moveTo):
    # #         contractLabels[mt] = self.contractLabels[mf]
    # #     self.contractLabels = contractLabels

    # def reArrange(self, labels):
    #     assert (len(labels) == len(self.labels)), "Error: number of tensor labels must be the same with original labels: get {} but {} needed".format(len(labels), len(self.labels))
    #     idx = self.getLabelIndices(labels)
    #     moveFrom = idx
    #     moveTo = list(range(len(moveFrom)))

    #     # self.moveContractLabels(moveFrom, moveTo)
    #     # if not self.lock:
    #     newLegs = [''] * self.dim
    #     for i in range(self.dim):
    #         newLegs[i] = self.legs[moveFrom[i]]
    #     self.a = self.xp.moveaxis(self.a, moveFrom, moveTo)
    #     self.shape = self.a.shape
    #     self.legs = newLegs
    #     # self.labels = deepcopy(labels)

    # def moveLabelsToFront(self, labs):
    #     #print('move label {}'.format(labs))
    #     #print('before move: {}'.format(self.labels))
    #     labelIndices = self.getLabelIndices(labs)
    #     # labelShape = self.shapeOfIndices(labelIndices)
    #     moveFrom = labelIndices
    #     moveTo = list(range(len(moveFrom)))
    #     # self.moveContractLabels(moveFrom, moveTo)
    #     #print('move from {} to {}'.format(moveFrom, moveTo))
    #     self.a = np.moveaxis(self.a, moveFrom, moveTo)
    #     # for lab in labs:
    #     #     self.labels.remove(lab)
    #     # self.labels = labs + self.labels
    #     newLegs = []
    #     for i in range(len(moveFrom)):
    #         newLegs.append(self.legs[moveFrom[i]])
    #     for leg in newLegs:
    #         self.legs.remove(leg)
    #     self.legs = newLegs + self.legs
    #     self.shape = self.a.shape
    #     #print('after move: {}'.format(self.labels))

    # # def selfContract(self, rows = None, cols = None):
    # #     if (rows is None):
    # #         rows = self.replicateLabels()
    # #     if (cols is None):
    # #         cols = self.replicateLabels()
    # #     data = self.toMatrix(rows = rows, cols = cols)
    # #     return np.trace(data)

    # # def selfSpectrum(self, rows = None, cols = None, d = None):
    # #     if (rows is None):
    # #         rows = self.replicateLabels()
    # #     if (cols is None):
    # #         cols = self.replicateLabels()
    # #     if (d is None):
    # #         d = (self.shape[0] ** 2)
    # #     data = self.toMatrix(rows = rows, cols = cols)
    # #     _, s, _ = np.linalg.svd(data)
    # #     #return np.trace(data)
    # #     return s[:d]
    # # def selfDiag(self, rows = None, cols = None):
    # #     if (rows is None):
    # #         rows = self.replicateLabels()
    # #     if (cols is None):
    # #         cols = self.replicateLabels()
    # #     data = self.toMatrix(rows = rows, cols = cols)
    # #     ret = sorted(np.diagonal(data), key = lambda x: np.abs(x))
    # #     ret.reverse()
    # #     return ret

    # def replicateLabels(self):
    #     labelSet = set(self.labels)
    #     return [lab for lab in labelSet if self.labels.count(lab) > 1]

    # # def selfContractLabels(self, lab):
    # #     assert (self.labels.count(lab) == 2), 'The label contracted should appear exactly twice in the tensor {}.\n'.format(self)
    # #     indicesForward = self.indexOfLabel(lab)
    # #     indicesBackward = self.indexOfLabel(lab, backward = True)
    # #     if (lab in Tensor.bondNameSet):
    # #         Tensor.bondNameSet.remove(lab)
    # #     return np.trace(self.a, axis1 = indicesForward, axis2 = indicesBackward), funcs.tupleRemoveByIndex(self.shape, [indicesForward, indicesBackward])
    
    # def copy(self):
    #     return Tensor(data = deepcopy(self.a), shape = deepcopy(self.shape), labels = deepcopy(self.labels), degreeOfFreedom = self.degreeOfFreedom)
    
    # def copyN(self, n):
    #     return tuple([self.copy() for i in range(n)])

    # # def toNDArray(self, labs):
    # #     labelIndices = self.getLabelIndices(labs)
    # #     initIndices = list(range(self.dim))
    # #     assert (set(labelIndices) == set(initIndices)), "Input label for toNDArray must have exactly same labels as tensor: input {} but have {}".format(labs, self.labels)
    # #     return np.copy(np.moveaxis(self.a, initIndices, labelIndices))

    # def outerProduct(self, labelList, newLabel):
    #     self.moveLabelsToFront(labelList)
    #     n = len(labelList)
    #     newShape = (-1, ) + self.shape[n:]
    #     self.a = np.reshape(self.a, newShape)
    #     self.shape = self.a.shape

    #     newDim = self.shape[0]
    #     self.legs = [Leg(self, newDim, newLabel)] + self.legs[n:]
        # for label in self.labels[:n]:
        #     del self.legs[label]
        # self.legs[newLabel] = Leg(self, newDim, newLabel)
        # self.labels = [newLabel] + self.labels[n:]
        # self.contractLabels = [''] + self.contractLabels[n:]

    # def swapLabel(self, lab1, lab2):
    #     idx = self.getLabelIndices([lab1, lab2])
    #     idx1, idx2 = tuple(idx)

    #     # self.contractLabels[idx2], self.contractLabels[idx1] = self.contractLabels[idx1], self.contractLabels[idx2]
    #     self.legs[idx2].name = lab1
    #     self.legs[idx1].name = lab2

    #     # self.legs[lab1], self.legs[lab2] = self.legs[lab2], self.legs[lab1]
    #     # if (self.legs[lab1].name == lab2):
    #     #     self.legs[lab1].name = lab1 
    #     # if (self.legs[lab2].name == lab1):
    #     #     self.legs[lab2].name = lab2

    # def normalize(self):
    #     normalFactor = np.sqrt(np.sum(np.abs(self.a) ** 2))
    #     self.a /= normalFactor
    #     return normalFactor

    # def sizeOfLabel(self, lab):
    #     # idx = self.indexOfLabel(lab)
    #     # return self.shape[idx]
    #     return self.getLeg(lab).dim

    # # def updateTensorData(self, data, labels):
    # #     idx = self.getLabelIndices(labels)
    # #     sp = tuple([self.shape[i] for i in idx])
    # #     self.a = np.reshape(data, sp)
    # #     self.labels = deepcopy(labels)
    # #     self.shape = sp

    # #     self.legs = dict([])
    # #     for label, dim in zip(self.labels, list(self.shape)):
    # #         self.legs = Leg(self, dim, label)

    # def norm(self):
    #     normalFactor = np.sqrt(np.sum(np.abs(self.a) ** 2))
    #     return normalFactor

    # # def lock(self):
    # #     self.locked = True
    # # def unlock(self):
    # #     self.locked = False

    # def getElementByNo(self, No):
    #     indexList = []
    #     for i in range(len(self.shape) - 1, -1, -1):
    #         indexList.append(No % self.shape[i])
    #         No //= self.shape[i]

    #     indexList.reverse()
    #     # print(tuple(indexList))
    #     # print('index = {}'.format(indexList))
    #     return self.a[tuple(indexList)]

    # def isSquareTensor(self):
    #     return funcs.compareLists(self.labels, ['l', 'u', 'r', 'd'])

    # def gaugeTransform(self, gauge):
    #     assert self.isSquareTensor(), "TensorBase.Transform only works for square tensors, but labels {} gotten.".format(self.labels)
    #     for contractLabel in ['l', 'u', 'r', 'd']:
    #         if not (contractLabel in gauge):
    #             continue
    #         self.moveLabelsToFront([contractLabel])
    #         self.a = np.tensordot(gauge[contractLabel], self.a, axes = ([1], [0]))
    #         self.shape = self.a.shape

    # def tensorTransform(self, mat, label):
    #     self.moveLabelsToFront([label])
    #     self.a = np.tensordot(mat.T, self.a, axes = ([1], [0]))
    #     self.shape = self.a.shape

    # def selfTrace(self):
    #     assert (self.isSquareTensor), 'Error: square tracing a tensor {} which is not a standard square tensor.'.format(self)
    #     return self.selfContract(rows = ['l', 'u'], cols = ['r', 'd'])

    # def getOriginalLabels(self, cLabels):
    #     return [self.labels[self.contractLabels.index(cLabel)] for cLabel in cLabels]
    # def addLink(self, linkTensor):
    #     self.linkedTensor.append(linkTensor)

