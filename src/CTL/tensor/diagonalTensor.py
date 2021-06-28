from CTL.tensorbase.tensorbase import TensorBase 
import CTL.funcs.funcs as funcs
import numpy as np 
from copy import deepcopy
from CTL.tensor.leg import Leg
from CTL.tensor.tensor import Tensor

class DiagonalTensor(Tensor):

    def deduceDimension(self, data, labels):
        # if the labels is given: then use labels
        # otherwise, if data is given(as an ndarray), then we return then len(data.shape)
        # otherwise, error
        if (labels is not None):
            return len(labels)
        elif (data is not None):
            # then data must be an numpy array
            return len(data.shape)
        else:
            raise ValueError(funcs.errorMessage(location = "DiagonalTensor.deduceDimension", err = "both data and labels are None."))
        

    def deduceData(self, data, labels, shape):
        # in Tensor: the "shape" has the highest priority
        # so if the shape is given here, it should be taken
        # however, if the shape is given as an integer: then we need to deduce the dimension

        # if shape exist: then according to shape(but dim may be deduced)
        # otherwise, if labels exist, then dim from labels, and l from data
        # otherwise, both dim and l from data
        funcName = "DiagonalTensor.deduceData"
        if (shape is not None):
            if (isinstance(shape, int)):
                dim = self.deduceDimension(data, labels)
                l = shape
            else:
                dim = len(shape)
                if (dim == 0) or (not funcs.checkAllEqual(shape)):
                    raise ValueError(funcs.errorMessage(location = funcName, err = "shape {} is not valid.".format(shape)))
                l = shape[0]
                # then we need to deduce dimension
            
            if (labels is not None) and (len(labels) != dim):
                raise ValueError(funcs.errorMessage(location = funcName, err = "number of labels is not the same as dim: {} expected but {} obtained.".format(dim, len(labels))))
            
            elif (data is not None):
                # data can be either shape, or an array of l
                if (len(data.shape) == 1):
                    if (data.shape[0] != l):
                        raise ValueError(funcs.errorMessage(location = funcName, err = "data length is not the same as length deduced from shape: {} expected but {} obtained.".format(l, data.shape[0])))
                
                elif (len(data.shape) != dim) or (data.shape[0] != l):
                    raise ValueError(funcs.errorMessage(location = funcName, err = "data shape is not correct: {} expected but {} obtained.".format(tuple([l] * dim), data.shape)))


            # shape is None, how to deduce shape?
        elif (labels is not None):
            dim = len(labels)
            if (data is None):
                raise ValueError(funcs.errorMessage(location = funcName, err = "cannot deduce data shape since data and shape are both None."))
            elif (len(data.shape) == 1):
                l = len(data)
            elif not funcs.checkAllEqual(data.shape):
                raise ValueError(funcs.errorMessage(location = funcName, err = "data.shape {} is not valid.".format(data.shape)))
            else:
                assert (len(data.shape) == dim), funcs.errorMessage(location = funcName, err = "dimension of data is not compatible with dimension deduced from labels: expect {} but {} is given.".format(dim, len(data.shape)))
                l = data.shape[0]
        
        else:
            # deduce from data.shape
            if (data is None):
                raise ValueError(funcs.errorMessage(location = funcName, err = "data, labes and shape are all None."))
            elif not funcs.checkAllEqual(data.shape):
                raise ValueError(funcs.errorMessage(location = funcName, err = "data.shape {} is not valid.".format(data.shape)))
            else:
                dim = len(data.shape)
                l = data.shape[0]

        print('l = {}, dim = {}'.format(l, dim))
            
        shape = tuple([l] * dim) 
        if (data is None):
            # default is identity
            data = self.xp.ones(l)
        elif (len(data.shape) == 1):
            data = self.xp.copy(data)
        else:
            data = self.xp.array([data[tuple([x] * dim)] for x in range(l)])

        if (labels is None):
            labels = self.generateLabels(dim)
        return data, labels, shape
            

    def __init__(self, shape = None, labels = None, data = None, degreeOfFreedom = None, name = None, legs = None):
        super().__init__(diagonalFlag = True)
        self.xp = np 

        data, labels, shape = self.deduceData(data, labels, shape)

        self.a = self.xp.copy(data)

        # functions of Tensor from here

        self.degreeOfFreedom = degreeOfFreedom
        self.name = name

        self._dim = len(shape)
        self._length = shape[0]

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
    def dim(self):
        return self._dim 
    
    @property 
    def shape(self):
        return tuple([self._length] * self._dim)
    
    @property 
    def labels(self):
        return [leg.name for leg in self.legs]
    
    @property 
    def chi(self):
        return self._length


    def __str__(self):
        if not (self.degreeOfFreedom is None):
            dofStr = ', degree of freedom = {}'.format(self.degreeOfFreedom)
        else:
            dofStr = ''
        if (self.name is not None):
            nameStr = self.name + ', ' 
        else:
            nameStr = ''
        return 'DiagonalTensor({}shape = {}, labels = {}{})'.format(nameStr, self.shape, self.labels, dofStr)

    def __repr__(self):
        if not (self.degreeOfFreedom is None):
            dofStr = ', degree of freedom = {}'.format(self.degreeOfFreedom)
        else:
            dofStr = ''
        if (self.name is not None):
            nameStr = self.name + ', ' 
        else:
            nameStr = ''
        return 'DiagonalTensor({}shape = {}, labels = {}{})\n'.format(nameStr, self.shape, self.labels, dofStr)

    def bondDimension(self):
        return self._length

    def toVector(self):
        funcs.deprecatedFuncWarning(funcName = "DiagonalTensor.toVector")
        return self.xp.copy(self.xp.ravel(self.a))
    
    def toMatrix(self, rows, cols):
        # print(rows, cols)
        # print(self.labels)
        # input two set of legs
        funcs.deprecatedFuncWarning(funcName = "DiagonalTensor.toMatrix")
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

        data = funcs.diagonalMatrix(self.a, self.dim)
        data = self.xp.reshape(data, (rowTotalSize, colTotalSize))
        return data

    def copy(self):
        return DiagonalTensor(data = self.xp.copy(self.a), degreeOfFreedom = self.degreeOfFreedom, name = self.name, labels = self.labels)
        # no copy of tensor legs, which may contain connection information

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

        self.legs = movedLegs + self.legs 
        # self.a = self.xp.moveaxis(self.a, moveFrom, moveTo)

    def outProduct(self, labelList, newLabel):
        raise TypeError(funcs.errorMessage(location = "DiagonalTensor.outProduct", err = "DiagonalTensor cannot perform outProduct, since the diagonal nature will be destroyed."))

    def norm(self):
        return self.xp.linalg.norm(self.a)

    def trace(self, rows = None, cols = None):
        return self.xp.sum(self.a)

    def single(self):
        # return the single value of this tensor
        # only works if shape == (,)
        assert self.shape == (), "Error: cannot get single value from tensor whose shape is not ()."
        return self.a

    def toTensor(self, labels):
        self.reArrange(labels)
        return funcs.diagonalMatrix(self.a, self.dim)
        
