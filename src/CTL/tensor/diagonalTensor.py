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

        if (data is not None) and (len(data.shape) != 1) and (labels is not None) and ((len(labels) != len(data.shape)) or (len(labels) == 0 and len(data.shape) == 1)):
            raise ValueError(funcs.errorMessage(location = "DiagonalTensor.deduceDimension", err = "data {} and labels {} are not compatible.".format(data, labels)))
        # what if len(labels) == 0, len(data.shape) == 1?

        if (labels is not None):
            return len(labels)
        elif (data is not None):
            # then data must be an numpy array
            return len(data.shape)
        else:
            raise ValueError(funcs.errorMessage(location = "DiagonalTensor.deduceDimension", err = "both data and labels are None."))
    
    # TODO: add the affect of "legs" to the deduction
    # the strategy is almost the same as Tensor
    # the only difference is that, when we have one integer as shape, and we have dimension: we can give the real shape by repeat for dim times
    
    # deduce strategy:
    # we want length and dim
    # priority for length: shape > data
    # priority for dim: shape > labels > data

    # 0. leg exist: the shape is already done
    # check if shape of leg is ok for diagonal tensor
    # if shape exist: check if shape is ok with shape of leg(integer / tuple)
    # if label exist: check if dimension of labels ok with legs
    # if data exist: ...

    # 1. shape exist: shape can be either an integer, or a n-element tuple
    # for int case: deduce dim from labels, then data
    # for tuple case: (length, data) is ready

    # then check labels: should be either None or len(labels) == dim
    # then check data: either None, length-element array, dim-dimensional tensor

    # 2. shape not exist: check labels for dim
    # then check data for dim(1d array, dim-d array with all equal shapes)
    # and generate l from shape of data

    # 3. labels not exist: check data for (dim, length)

    def checkLegsDiagonalCompatible(self, legs):
        if (len(legs) == 0):
            return True 
        l = legs[0].dim 
        for leg in legs:
            if (leg.dim != l):
                return False 
        return True
    def checkShapeDiagonalCompatible(self, shape):
        if (len(shape) == 0):
            return True 
        l = shape[0]
        for dim in shape:
            if (dim != l):
                return False 
        return True

    def checkLegsShapeCompatible(self, legs, shape):
        if (shape is None):
            return True 
        if (isinstance(shape, int)):
            shape = tuple([shape] * len(legs))
        if (isinstance(shape, list) or isinstance(shape, tuple)):
            shapeList = list(shape)
            if (len(shapeList) != len(legs)):
                return False 
            for dim, leg in zip(shapeList, legs):
                if (dim != leg.dim):
                    return False 
            return True
        else:
            return False
    
    def checkShapeDataCompatible(self, shape, data):
        # we know shape, and want to see if data is ok
        if (data is None):
            return True 
        if (isinstance(shape, int)):
            shape = tuple([shape] * len(data.shape))
        return ((len(data.shape) == 1) and (len(shape) > 0) and (len(data) == shape[0])) or (funcs.tupleProduct(data.shape) == funcs.tupleProduct(shape))

    def generateData(self, shape, data, isTensorLike):
        if (isTensorLike):
            return None
        # print('generating data for data = {}'.format(data))
        if (data is None):
            data = self.xp.ones(shape[0])
        # otherwise, data can be 1D-array, or ndarray
        elif (len(data.shape) == 1):
            data = self.xp.copy(data)
        else:
            l, dim = len(shape), shape[0]
            # print('dim = {}, l = {}'.format(dim, l))
            # print(self.xp.diag_indices(dim, l))
            data = self.xp.copy(data[self.xp.diag_indices(dim, l)])
        return data

    def deduction(self, legs, data, labels, shape, isTensorLike = False):
        # in Tensor: the "shape" has the highest priority
        # so if the shape is given here, it should be taken
        # however, if the shape is given as an integer: then we need to deduce the dimension

        # if shape exist: then according to shape(but dim may be deduced)
        # otherwise, if labels exist, then dim from labels, and l from data
        # otherwise, both dim and l from data
        funcName = "DiagonalTensor.deduction"

        if (legs is not None):

            if (not self.checkLegsDiagonalCompatible(legs = legs)):
                raise ValueError(funcs.errorMessage('legs {} cannot be considered as legs for diagonal tensor.'.format(legs), location = funcName))

            if (not self.checkLegsLabelsCompatible(legs = legs, labels = labels)):
                raise ValueError(funcs.errorMessage('labels {} is not compatible with legs {}'.format(labels, legs), location = funcName))
            if (labels is None):
                labels = [leg.name for leg in legs]

            if (not self.checkLegsShapeCompatible(legs = legs, shape = shape)):
                raise ValueError(funcs.errorMessage('shape {} is not compatible with legs {}'.format(shape, legs), location = funcName))
            if (shape is None) or (isinstance(shape, int)):
                shape = tuple([leg.dim for leg in legs]) 

            if (not self.checkShapeDataCompatible(shape = shape, data = data)):
                raise ValueError(funcs.errorMessage('data shape {} is not compatible with required shape {}'.format(data.shape, shape), location = funcName))

        elif (shape is not None):

            if (isinstance(shape, int)):
                dim = self.deduceDimension(data = data, labels = labels)
                shape = tuple([shape] * dim)
            
            if (not self.checkShapeDiagonalCompatible(shape = shape)):
                raise ValueError(funcs.errorMessage('shape {} cannot be considered as shape for diagonal tensor.'.format(shape), location = funcName))

            if (not self.checkShapeLabelsCompatible(shape = shape, labels = labels)):
                raise ValueError(funcs.errorMessage('labels {} is not compatible with required shape {}'.format(labels, shape), location = funcName))
            if (labels is None):
                labels = self.generateLabels(len(shape))
            
            if (not self.checkShapeDataCompatible(shape = shape, data = data)):
                raise ValueError(funcs.errorMessage('data shape {} is not compatible with required shape {}'.format(data.shape, shape), location = funcName))

        elif (data is not None):
            # legs, shape are both None
            shape = data.shape 
            if (not self.checkShapeDiagonalCompatible(shape = shape)):
                raise ValueError(funcs.errorMessage('data shape {} cannot be considered as shape for diagonal tensor.'.format(shape), location = funcName))
            
            dim = self.deduceDimension(data = data, labels = labels)
            if (len(shape) == 1) and (dim > 1):
                shape = tuple([shape[0]] * dim)

            if (not self.checkShapeLabelsCompatible(shape = shape, labels = labels)):
                raise ValueError(funcs.errorMessage('labels {} is not compatible with required shape {}'.format(labels, shape), location = funcName))
            if (labels is None):
                labels = self.generateLabels(len(shape))

        else:
            raise ValueError(funcs.errorMessage("Tensor() cannot accept parameters where legs, shape and data being None simultaneously.", location = funcName))

        # elif (shape is not None):
        #     if (isinstance(shape, int)):
        #         dim = self.deduceDimension(data, labels)
        #         l = shape
        #     else:
        #         dim = len(shape)
        #         if (dim == 0) or (not funcs.checkAllEqual(shape)):
        #             raise ValueError(funcs.errorMessage(location = funcName, err = "shape {} is not valid.".format(shape)))
        #         l = shape[0]
        #         # then we need to deduce dimension
            
        #     if (labels is not None) and (len(labels) != dim):
        #         raise ValueError(funcs.errorMessage(location = funcName, err = "number of labels is not the same as dim: {} expected but {} obtained.".format(dim, len(labels))))
            
        #     elif (data is not None):
        #         # data can be either shape, or an array of l
        #         if (len(data.shape) == 1):
        #             if (data.shape[0] != l):
        #                 raise ValueError(funcs.errorMessage(location = funcName, err = "data length is not the same as length deduced from shape: {} expected but {} obtained.".format(l, data.shape[0])))
                
        #         elif (len(data.shape) != dim) or (data.shape != tuple([l] * dim)):
        #             raise ValueError(funcs.errorMessage(location = funcName, err = "data shape is not correct: {} expected but {} obtained.".format(tuple([l] * dim), data.shape)))


        #     # shape is None, how to deduce shape?
        # elif (labels is not None):
        #     dim = len(labels)
        #     if (data is None):
        #         raise ValueError(funcs.errorMessage(location = funcName, err = "cannot deduce data shape since data and shape are both None."))
        #     elif (len(data.shape) == 1):
        #         l = len(data)
        #     elif not funcs.checkAllEqual(data.shape):
        #         raise ValueError(funcs.errorMessage(location = funcName, err = "data.shape {} is not valid.".format(data.shape)))
        #     else:
        #         if (len(data.shape) != dim):
        #             raise ValueError(funcs.errorMessage(location = funcName, err = "dimension of data is not compatible with dimension deduced from labels: expect {} but {} is given.".format(dim, len(data.shape))))
        #         l = data.shape[0]
        
        # else:
        #     # deduce from data.shape
        #     if (data is None):
        #         raise ValueError(funcs.errorMessage(location = funcName, err = "data, labes and shape are all None."))
        #     elif not funcs.checkAllEqual(data.shape):
        #         raise ValueError(funcs.errorMessage(location = funcName, err = "data.shape {} is not valid.".format(data.shape)))
        #     else:
        #         dim = len(data.shape)
        #         l = data.shape[0]

        # print('l = {}, dim = {}'.format(l, dim))
            
        # shape = tuple([l] * dim) 

        data = self.generateData(shape = shape, data = data, isTensorLike = isTensorLike)

        # if (tensorLikeFlag):
        #     data = None
        # elif (data is None):
        #     # default is identity
        #     data = self.xp.ones(l)
        # elif (len(data.shape) == 1):
        #     data = self.xp.copy(data)
        # else:
        #     data = self.xp.array([data[tuple([x] * dim)] for x in range(l)])

        # must be a copy of original "data" if exist

        # if (labels is None):
        #     labels = self.generateLabels(dim)
        
        if (legs is None):
            legs = []
            for label, dim in zip(labels, list(shape)):
                legs.append(Leg(self, dim, label))

        else:
            for leg in legs:
                leg.tensor = self

        return legs, data, labels, shape

        
            

    def __init__(self, shape = None, labels = None, data = None, degreeOfFreedom = None, name = None, legs = None, tensorLikeFlag = False):
        super().__init__(diagonalFlag = True, tensorLikeFlag = tensorLikeFlag)

        legs, data, labels, shape = self.deduction(legs = legs, data = data, labels = labels, shape = shape, isTensorLike = tensorLikeFlag)

        self.a = data
        self.legs = legs
        self.totalSize = funcs.tupleProduct(shape)

        # functions of Tensor from here

        self.degreeOfFreedom = degreeOfFreedom
        self.name = name

        self._dim = len(shape)
        self._length = shape[0]

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
        if (self.tensorLikeFlag):
            objectStr = 'DiagonalTensorLike'
        else:
            objectStr = 'DiagonalTensor'
        if not (self.degreeOfFreedom is None):
            dofStr = ', degree of freedom = {}'.format(self.degreeOfFreedom)
        else:
            dofStr = ''
        if (self.name is not None):
            nameStr = self.name + ', ' 
        else:
            nameStr = ''
        return '{}({}shape = {}, labels = {}{})'.format(objectStr, nameStr, self.shape, self.labels, dofStr)

    def __repr__(self):
        if (self.tensorLikeFlag):
            objectStr = 'DiagonalTensorLike'
        else:
            objectStr = 'DiagonalTensor'
        if not (self.degreeOfFreedom is None):
            dofStr = ', degree of freedom = {}'.format(self.degreeOfFreedom)
        else:
            dofStr = ''
        if (self.name is not None):
            nameStr = self.name + ', ' 
        else:
            nameStr = ''
        return '{}({}shape = {}, labels = {}{})'.format(objectStr, nameStr, self.shape, self.labels, dofStr)

    def bondDimension(self):
        return self._length

    def moveLegsToFront(self, legs):
        moveFrom = []
        moveTo = []
        currIdx = 0
        movedLegs = legs
        for currLeg in legs:
            for i, leg in enumerate(self.legs):
                if (leg == currLeg):
                    moveFrom.append(i)
                    moveTo.append(currIdx)
                    currIdx += 1
                    break

        for leg in movedLegs:
            self.legs.remove(leg)
        
        # print(moveFrom, moveTo)
        # print(labelList)
        # print(self.labels)
        self.legs = movedLegs + self.legs 
        # self.a = self.xp.moveaxis(self.a, moveFrom, moveTo)

    def toVector(self):
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike cannot be transferred to vector since no data contained.', 'DiagonalTensor.toVector')
        funcs.deprecatedFuncWarning(funcName = "DiagonalTensor.toVector", deprecateMessage = "This will return a vector corresponding to the diagonal of tensor instead of the complete tensor.")
        return self.xp.copy(self.xp.ravel(self.a))
    
    def toMatrix(self, rows, cols):
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike cannot be transferred to matrix since no data contained.', 'DiagonalTensor.toMatrix')
        # print(rows, cols)
        # print(self.labels)
        # input two set of legs
        funcs.deprecatedFuncWarning(funcName = "DiagonalTensor.toMatrix", deprecateMessage = "Diagonal tensors should be used in a better way for linear algebra calculation rather than be made into a matrix.")
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
        return DiagonalTensor(data = self.a, shape = self.shape, degreeOfFreedom = self.degreeOfFreedom, name = self.name, labels = self.labels, tensorLikeFlag = self.tensorLikeFlag)
        # no copy of tensor legs, which may contain connection information
    def toTensorLike(self):
        if (self.tensorLikeFlag):
            return self.copy()
        else:
            return DiagonalTensor(data = None, degreeOfFreedom = self.degreeOfFreedom, name = self.name, labels = self.labels, shape = self.shape, tensorLikeFlag = True)

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
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike do not have norm since no data contained.', 'DiagonalTensor.norm')
        return self.xp.linalg.norm(self.a)

    def trace(self, rows = None, cols = None):
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike do not have trace since no data contained.', 'DiagonalTensor.trace')
        return self.xp.sum(self.a)

    def single(self):
        # return the single value of this tensor
        # only works if shape == (,)
        # assert self.shape == (), "Error: cannot get single value from tensor whose shape is not ()."
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike cannot be transferred to single value since no data contained.', 'DiagonalTensor.single')
        assert self._length == 1, "Error: cannot get single value from diagTensor whose length is not (1,)."
        return self.a

    def toTensor(self, labels = None):
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike cannot be transferred to tensor since no data contained.', 'DiagonalTensor.toTensor')
        if (labels is not None):
            self.reArrange(labels)
        return funcs.diagonalMatrix(self.a, self.dim)

    def typeName(self):
        if (self.tensorLikeFlag):
            return "DiagonalTensorLike"
        else:
            return "DiagonalTensor"
        
