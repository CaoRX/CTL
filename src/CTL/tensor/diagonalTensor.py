from CTL.tensorbase.tensorbase import TensorBase 
import CTL.funcs.funcs as funcs
import numpy as np 
from copy import deepcopy
from CTL.tensor.leg import Leg
from CTL.tensor.tensor import Tensor
import warnings

class DiagonalTensor(Tensor):
    """
    The class for diagonal tensors, inheriting from Tensor
    1. A data tensor as 1D-array: the elements on the main diagonal;
    2. A set of legs, corresponding to each dimension of the tensor.
    3. Other information(degree of freedom, total element number, ...)

    Diagonal Tensors: a tensor with only non-zero elements on its main diagonal, e.g., for a 3-dimensional diagonal tensor A, only A_{iii} is non-zero, while A_{123} must be zero.

    This class is also used for DiagonalTensorLike, an object that behaves almost the same as DiagonalTensor, but without data.

    In the following docstrings we will take the number of elements as $n$, the dimension as $d$, and then make some statements on the time efficiency for some functions.

    In other part of docstrings, we will not talk about Tensor and DiagonalTensor separately except for special cases.

    Parameters
    ----------
    shape : None or tuple of int, optional
        The expected shape of the tensor.
    labels : None or tuple of str, optional
        The labels to be put for each dimension, if None then automatically generated from lower case letters.
    data : None or ndarray or 1D-array of float, optional
        The data in the tensor. 
        If None and the data is needed(not TensorLike), then generated as np.random.random_sample. 
        If shape is given, data does not need to have the same shape as "shape", but the number of elements should be the same.
        If 1D-array, then taken as the diagonal elements, can be used for diagonal tensors of any rank.
    degreeOfFreedom : None or int, optional
        Local degree of freedom for this tensor.
    name : None or str, optional
        The name of the tensor to create.
    legs : None or list of Leg, optional
        The legs of this tensor. If None, then automatically generated.
    diagonalFlag : bool, default False
        Whether this tensor is diagonal tensor or not. Diagonal tensors can behave better in efficiency for tensor contractions, so we deal with them with child class DiagonalTensor, check the details in CTL.tensor.diagonalTensor.
    tensorLikeFlag : bool, default False
        If True, then the tensor is a "TensorLike": will not contain any data, but behave just like a tensor.
    xp : object, default numpy
		The numpy-like library for numeric functions.

    Attributes
    ----------
    tensorLikeFlag : bool
        Whether the tensor is a "TensorLike".
    xp : object
        The numpy-like library for numeric functions.
    diagonalFlag : bool
        Whether the tensor is a "DiagonalTensor"
    totalSize : int
        Total number of components in this tensor.
    degreeOfFreedom : int
        Number of local degree of freedom. E.g. for Ising Tensor around one spin, it can be 1. 
    name : None or str
        The name of the tensor.
    legs : list of Leg
        The legs from this tensor, can be "attracted" to another leg to form a bond. If not so, then it is a free leg.
    a : ndarray of float
        The data of the tensor.

    Notes
    -----
    Please note shape, labels, data and legs: although they are all optional, they need to contain enough(and not contradictory) information for deduce the shape, labels, data and legs for the tensor, the deduction strategy is described below:

    For labels: priority is legs = labels, default: auto-generated in order from lowercase letters.

    For shape: priority is legs = shape > data.

    For legs: priority is legs, default: auto-generated with labels and shape.

    For data: priority is data.reshape(shape), default: np.random.random_sample(shape).

    ("For property A, priority is B > C = D > E, default: F" means, A can be deduced from B, C, D, E, so we consider from high priority to low priority. If B exist, then we take the deduced value from B, and change C, D, E if they in some sense compatible with B. Otherwise consider C & D. For values of the same priority, if both of them are provided, then they should be the same. If none of B, C, D, E can deduce A, then generate A with F.)

    "checkXXXYYYCompatible" functions will do the above checkings to make the information in the same priority compatible with each other.
    """

    def deduceDimension(self, data, labels):
        """
        Deduce the dimension of current diagonal tensor from data and labels. 

        Parameters
        ----------
        data : None or 1D array or ndarray
            The data to be put in the diagonal tensor.
        labels : None or list of Leg
            The labels to be added to the legs of this tensor.
        
        Returns
        -------
        int
            The dimension of the current tensor.
        """
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
        """
        Check whether the shape from legs can form a diagonal tensor, with all the indices have the same dimension.

        Parameters
        ----------
        legs : list of Leg
            Legs of the tensor that already existed before creating the tensor.

        Returns
        -------
        bool
            Whether the legs can form a diagonal tensor.
        """
        if (len(legs) == 0):
            return True 
        l = legs[0].dim 
        for leg in legs:
            if (leg.dim != l):
                return False 
        return True
    def checkShapeDiagonalCompatible(self, shape):
        """
        Check whether the shape can form a diagonal tensor, with all the indices have the same dimension.

        Parameters
        ----------
        shape : tuple of int
            Shape of the tensor that already existed before creating the tensor.

        Returns
        -------
        bool
            Whether the legs can form a diagonal tensor.
        """
        if (len(shape) == 0):
            return True 
        l = shape[0]
        for dim in shape:
            if (dim != l):
                return False 
        return True

    def checkLegsShapeCompatible(self, legs, shape):
        """
        For information, check Tensor.checkLegsShapeCompatible.
        """
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
        """
        For information, check Tensor.checkShapeDataCompatible.
        """
        # we know shape, and want to see if data is ok
        if (data is None):
            return True 
        if (isinstance(shape, int)):
            shape = tuple([shape] * len(data.shape))
        return ((len(data.shape) == 1) and (len(shape) > 0) and (len(data) == shape[0])) or (funcs.tupleProduct(data.shape) == funcs.tupleProduct(shape))

    def generateData(self, shape, data, isTensorLike):
        """
        For information, check Tensor.generateData.

        Returns
        -------
        1D-array of float
            The data to be saved in this diagonal tensor.
        """
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
        """
        For more information, check Tensor.deduction
        """
        # in Tensor: the "shape" has the highest priority
        # so if the shape is given here, it should be taken
        # however, if the shape is given as an integer: then we need to deduce the dimension

        # if shape exist: then according to shape(but dim may be deduced)
        # otherwise, if labels exist, then dim from labels, and l from data
        # otherwise, both dim and l from data
        funcName = "DiagonalTensor.deduction"

        # first, consider scalar case
        if (legs is None) and (labels is None) and (shape == () or ((data is not None) and (data.shape == ()))):
            if (data is None) and (not isTensorLike):
                data = np.array(1.0)
            return [], data, [], () # scalar
            

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

        
            

    def __init__(self, shape = None, labels = None, data = None, degreeOfFreedom = None, name = None, legs = None, tensorLikeFlag = False, xp = np, dtype = np.float64):
        super().__init__(diagonalFlag = True, tensorLikeFlag = tensorLikeFlag, xp = xp, dtype = dtype)

        legs, data, labels, shape = self.deduction(legs = legs, data = data, labels = labels, shape = shape, isTensorLike = tensorLikeFlag)

        self.a = data
        self.legs = legs
        # self.totalSize = funcs.tupleProduct(shape)

        # functions of Tensor from here

        self.degreeOfFreedom = degreeOfFreedom
        self.name = name

        # self._dim = len(shape)
        
        if shape == ():
            self._length = 1
        else:
            self._length = shape[0]

    @property 
    def dim(self):
        return len(self.legs)
    
    @property 
    def shape(self):
        return tuple([self._length] * self.dim)
    
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
        """
        The bond dimension of the current diagonal tensor: it is the same over all dimensions.

        Returns
        -------
        int
            The dimension for each index.
        """
        return self._length

    def moveLegsToFront(self, legs):
        """
        Change the orders of legs: move a given set of legs to the front while not modifying the relative order of other legs. Use self.xp.moveaxis to modify the data if this is not a TensorLike object.

        In fact make nothing difference for diagonal tensor: for Tensor this function will change the order of indices of data, but for diagonal tensor it is only a virtual change of legs.

        Parameters
        ----------
        legs : list of Leg
            The set of legs to be put at front.

        """
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
        """
        Deprecated

        Make a vector according to the diagonal elements.

        Deprecated since this behavior is different from Tensor, which will return a flattened data of ndarray. However, if we return the ndarray, this is usually useless for diagonal tensor and may generate an issue of CPU time. 

        To obtain the data, DiagonalTensor.a is enough.

        Returns
        -------
        1D ndarray of float
            A vector contains diagonal elements of the diagonal tensor.

        """
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike cannot be transferred to vector since no data contained.', 'DiagonalTensor.toVector')
        funcs.deprecatedFuncWarning(funcName = "DiagonalTensor.toVector", deprecateMessage = "This will return a vector corresponding to the diagonal of tensor instead of the complete tensor.")
        return self.xp.copy(self.xp.ravel(self.a))
    
    def toMatrix(self, rows, cols):
        """
        Deprecated

        Make a matrix of the data of this diagonal tensor, given the labels or legs of rows and cols. 

        Deprecated since this function is time comsuming(O(n^d)), and for most of the cases there are much better ways to use the data rather than making a matrix. For details, see CTL.tensor.contract for more information.

        Parameters
        ----------
        rows : None or list of str or list of Leg
            The legs for the rows of the matrix. If None, deducted from cols.
        cols : None or list of str or list of Leg
            The legs for the cols of the matrix. If None, deducted from rows.

        Returns
        -------
        2D ndarray of float
            The data of this tensor, in the form of (rows, cols).
        """
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

        data = funcs.diagonalNDTensor(self.a, self.dim)
        data = self.xp.reshape(data, (rowTotalSize, colTotalSize))
        return data

    def copy(self):
        """
        Make a copy of current diagonal tensor, without copy the legs. For more information, refere to Tensor.copy

        Returns
        -------
        DiagonalTensor
            A copy of the current diagonal tensor, all the information can be copied is contained.
        """
        return DiagonalTensor(data = self.a, shape = self.shape, degreeOfFreedom = self.degreeOfFreedom, name = self.name, labels = self.labels, tensorLikeFlag = self.tensorLikeFlag)
        # no copy of tensor legs, which may contain connection information
    def toTensorLike(self):
        """
        Make a copy of current tensor, without copying the legs. This function works almost like self.copy(), but without copying the data.

        Returns
        -------
        DiagonalTensor
            A DiagonalTensorLike of the current tensor, all the information can be copied is contained except legs and data.
        """
        if (self.tensorLikeFlag):
            return self.copy()
        else:
            return DiagonalTensor(data = None, degreeOfFreedom = self.degreeOfFreedom, name = self.name, labels = self.labels, shape = self.shape, tensorLikeFlag = True)

    def moveLabelsToFront(self, labelList):
        """
        Change the orders of legs: move a given set of labels to the front. For details, check "self.moveLegsToFront".

        Parameters
        ----------
        labelList : list of str
            The set of labels to be put at front.

        """
        legs = self.getLegsByLabel(labelList)
        self.moveLegsToFront(legs)
        # legs = [self.getLeg(label) for label in labelList]
        # self.moveLegsToFront(legs)

        # moveFrom = []
        # moveTo = []
        # currIdx = 0
        # movedLegs = []
        # for label in labelList:
        #     for i, leg in enumerate(self.legs):
        #         if (leg.name == label):
        #             moveFrom.append(i)
        #             moveTo.append(currIdx)
        #             currIdx += 1
        #             movedLegs.append(leg)
        #             break

        # for leg in movedLegs:
        #     self.legs.remove(leg)

        # self.legs = movedLegs + self.legs 
        # self.a = self.xp.moveaxis(self.a, moveFrom, moveTo)

    def outProduct(self, labelList, newLabel):
        """
        Deprecated

        Comment
        -------
        The outer product will destroy the shape of diagonal tensor: we cannot easily combine several legs if it is a full diagonal tensor, so a TypeError will be raised.
        """
        raise TypeError(funcs.errorMessage(location = "DiagonalTensor.outProduct", err = "DiagonalTensor cannot perform outProduct, since the diagonal nature will be destroyed."))

    def norm(self):
        """
        Norm of the current tensor. O(n).

        Returns
        -------
        float
            The norm of data.
        """
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike do not have norm since no data contained.', 'DiagonalTensor.norm')
        return self.xp.linalg.norm(self.a)

    def trace(self, rows = None, cols = None):
        """
        Trace of the current diagonal tensor. To not destroy the property for the diagonal tensors, this function can only be used to calculate the global trace on the main diagonal.

        Parameters
        ----------
        rows, cols: None
            Only set to be compatible with the usage for Tensor
        
        Returns
        -------
        float
            The trace of the matrix generated by given cols and rows.
        """
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike do not have trace since no data contained.', 'DiagonalTensor.trace')
        return self.xp.sum(self.a)

    def single(self):
        """
        Generate a single value from a tensor. 

        Note the difference between this and Tensor.single(): in Tensor object, the data are saved as ndarray, so for single value it must be a 0-d array, in other words, a single number. 

        However, for DiagonalTensor: in all cases the data are saved as 1D-array, so we need to first decide whether it can be transferred to a single number, and then return the lowest index.

        Returns
        -------
        float
            A single value of this tensor.

        """
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike cannot be transferred to single value since no data contained.', 'DiagonalTensor.single')
        assert self._length == 1, "Error: cannot get single value from diagTensor whose length is not (1,)."
        assert self.shape == (), "Error: cannot get single value from tensor whose shape is not ()."
        return self.a[()]

    def toTensor(self, labels = None):
        """
        Return a ndarray of this tensor. Since the current tensor object only saves the main diagonal, the tensor itself may be much larger, so this is not recommended and not used in any of the internal functions.

        Parameters
        ----------
        labels : None or list of str
            The order of labels for the output tensor. Note that if labels is None, the order of legs is not fixed, may differ from time to time.
        
        Returns
        -------
        ndarray of float
            The data of the tensor, order of legs are given by the labels.
        """
        assert (not self.tensorLikeFlag), funcs.errorMessage('DiagonalTensorLike cannot be transferred to tensor since no data contained.', 'DiagonalTensor.toTensor')
        if (labels is not None):
            self.reArrange(labels)
        return funcs.diagonalNDTensor(self.a, self.dim)

    def sumOutLeg(self, leg):
        """
        Sum out one leg to make a (D - 1)-dimensional tensor. Give a warning(and do nothing) if leg is not one of the current tensor, and give a warning if leg is connected to some bond(not free).

        Parameters
        ----------
        leg : Leg
            The leg to be summed out.

        """
        if not (leg in self.legs):
            warnings.warn(funcs.warningMessage("leg {} is not in tensor {}, do nothing.".format(leg, self), location = 'Tensor.sumOutLeg'), RuntimeWarning)
            return
        if leg.bond is not None:
            warnings.warn(funcs.warningMessage("leg {} to be summed out is connected to bond {}.".format(leg, leg.bond), location = 'Tensor.sumOutLeg'), RuntimeWarning)
        
        idx = self.legs.index(leg)
        # self.a = self.xp.sum(self.a, axis = idx)
        self.legs = self.legs[:idx] + self.legs[(idx + 1):]
        if (len(self.legs) == 0):
            # not a diagonal tensor, since the last sum will give a single value
            self.a = np.array(np.sum(self.a))
            self._length = 1

    def typeName(self):
        """
        The type of the current class.

        Returns
        -------
        {"DiagonalTensor", "DiagonalTensorLike"}
        """
        if (self.tensorLikeFlag):
            return "DiagonalTensorLike"
        else:
            return "DiagonalTensor"
        
