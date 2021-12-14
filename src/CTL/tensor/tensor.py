from CTL.tensorbase.tensorbase import TensorBase 
import CTL.funcs.funcs as funcs
# import numpy as np 
import CTL.funcs.xplib as xplib
from copy import deepcopy
from CTL.tensor.leg import Leg
import warnings

class Tensor(TensorBase):
    """
    The main class of Tensor. The class is organized as:
    1. A data tensor as ndarray;
    2. A set of legs, corresponding to each dimension of the tensor.
    3. Other information(degree of freedom, total element number, ...)

    This class is also used for TensorLike, an object that behaves almost the same as Tensor, but without data.

    Parameters
    ----------
    shape : None or tuple of int, optional
        The expected shape of the tensor.
    labels : None or tuple of str, optional
        The labels to be put for each dimension, if None then automatically generated from lower case letters.
    data : None or ndarray of float, optional
        The data in the tensor. 
        If None and the data is needed(not TensorLike), then generated as np.random.random_sample. 
        If shape is given, data does not need to have the same shape as "shape", but the number of elements should be the same.
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
    dtype : type, default numpy.float64
        The type of data elements in tensors.

    Notes
    -----
    Please note shape, labels, data and legs: although they are all optional, they need to contain enough(and not contradictory) information for deduce the shape, labels, data and legs for the tensor, the deduction strategy is described below:

    For labels: priority is legs = labels, default: auto-generated in order from lowercase letters.

    For shape: priority is legs = shape > data.

    For legs: priority is legs, default: auto-generated with labels and shape.

    For data: priority is data.reshape(shape), default: np.random.random_sample(shape).

    For data type(dtype): data.dtype > self.dtype

    ("For property A, priority is B > C = D > E, default: F" means, A can be deduced from B, C, D, E, so we consider from high priority to low priority. If B exist, then we take the deduced value from B, and change C, D, E if they in some sense compatible with B. Otherwise consider C & D. For values of the same priority, if both of them are provided, then they should be the same. If none of B, C, D, E can deduce A, then generate A with F.)

    "checkXXXYYYCompatible" functions will do the above checkings to make the information in the same priority compatible with each other.
    """

    def isFloatTensor(self):
        """
        Decide whether the base data type of the tensor is float

        Returns
        -------
        bool
            Whether the dtype of the tensor is float(float16, float32, float64, ...) or not.
        """
        return xplib.xp.issubdtype(self.dtype, xplib.xp.floating)
    
    def deduceDataType(self, data, dtype):
        if data is None:
            return dtype 
        else:
            return data.dtype

    def __init__(self, shape = None, labels = None, data = None, degreeOfFreedom = None, name = None, legs = None, diagonalFlag = False, tensorLikeFlag = False, dtype = xplib.xp.float64):
        super().__init__(None)

        # only data needs a copy
        # labels and shape: they are virtual, will be saved in legs
        self.dtype = self.deduceDataType(data = data, dtype = dtype)
        self.tensorLikeFlag = tensorLikeFlag
        if (diagonalFlag):
            self.diagonalFlag = True 
            return 
        else:
            self.diagonalFlag = False

        legs, shape, labels, data = self.deduction(legs = legs, shape = shape, labels = labels, data = data, isTensorLike = tensorLikeFlag)
        
        # self.totalSize = funcs.tupleProduct(shape)
        self.degreeOfFreedom = degreeOfFreedom
        self.name = name

        self.legs = legs 
        self.a = data

    # def __init__(self):
    #     pass
    # attributes: xp, totalSize, labels, contractLabels, degreeOfFreedom, a

    # bondNameSet = set([])

    # we want to update the deduction of the shape of tensor
    # there are problems now(July 7th, 2021) that:
    # if we give legs with different shape from the shape(or data), it will keep the data while Tensor.shape gives different results from Tensor.a.shape
    # to solve this problem: we need to change the priority of deduction to the shape
    
    # deduce strategy:
    # we want shape, and labels
    # we have legs, shape, labels, data
    # priority for shape: legs > shape > data
    # priority for labels: legs > labels

    # 1. legs exist: 
    # if labels exist: check the length and content of labels with legs
    # if shape exist: check whether shape == tuple([leg.dim for leg in legs])
    # if data exist: check whehter data.shape == tuple([leg.dim for leg in legs]) (if not, but the total size equal, we transfer data to the given shape)

    # 2. legs not exist, shape exist
    # if data exist, check the total number of components of data equal to shape, otherwise random
    # if labels exist: check the number of labels equal to dimension, otherwise auto-generate
    
    # 3. legs not exist, shape not exist
    # if data exist: generate shape according to data, and auto-generate legs

    # now we can add the tensor-like property
    # we still deduce the legs, shape, labels, data, but without real data

    def checkLegsLabelsCompatible(self, legs, labels):
        """
        Check whether labels is compatible with legs. For more information, check "Notes" of comments for Tensor.

        Parameters
        ----------
        legs : list of Leg
            Legs of the tensor that already existed before creating the tensor.
        labels : None or list of str
            The labels to be added to the legs of this tensor.
        
        Returns
        -------
        bool
            Whether the legs and labels are compatible.
        """
        # we know that legs is not None
        # so labels should be either None, or a list that corresponding to legs
        if (labels is None):
            return True 
        if (isinstance(labels, list) or isinstance(labels, tuple)):
            labelList = list(labels)
            if (len(labelList) != len(legs)):
                return False 
            for label, leg in zip(labelList, legs):
                if (label != leg.name):
                    return False 
            return True
        else:
            return False

    def checkLegsShapeCompatible(self, legs, shape):
        """
        Check whether shape is compatible with legs. For more information, check "Notes" of comments for Tensor.

        Parameters
        ----------
        legs : list of Leg
            Legs of the tensor that already existed before creating the tensor.
        shape : None or tuple of int
            The expected shape of the tensor.
        
        Returns
        -------
        bool
            Whether the legs and shape are compatible.
        """
        if (shape is None):
            return True 
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
        Check whether data is compatible with shape. For more information, check "Notes" of comments for Tensor.

        Parameters
        ----------
        shape : tuple of int
            The expected shape of the tensor.
        data : None or ndarray of float
            The data to be put into the tensor.
        
        Returns
        -------
        bool
            Whether the shape and data are compatible.
        """
        # we know shape, and want to see if data is ok
        if (data is None):
            return True 
        return (funcs.tupleProduct(data.shape) == funcs.tupleProduct(shape))

    def checkShapeLabelsCompatible(self, shape, labels):
        """
        Check whether labels is compatible with shape. For more information, check "Notes" of comments for Tensor.

        Parameters
        ----------
        shape : tuple of int
            The expected shape of the tensor.
        labels : None or list of str
            The labels to be added to the legs of this tensor.
        
        Returns
        -------
        bool
            Whether the shape and labels are compatible.
        """
        if (labels is None):
            return True 
        return len(labels) == len(shape)

    def generateData(self, shape, data, isTensorLike):
        """
        Generate the data for this tensor. None for TensorLike, if data is None then randomly generated, otherwise, reshape to the given shape and return a copy.

        Parameters
        ----------
        shape : tuple of int
            Expected shape for the tensor.
        data : None or ndarray of float
            The data to be put in the tensor, None for randomly generated.
        isTensorLike : bool
            Whether we are working for a TensorLike object: if True, then data is None.
        
        Returns
        -------
        None or ndarray of float
            For TensorLike, return None.
            If data exists, then reshape it to shape, and return its copy.
            If data is None, then randomly generate a tensor of shape.
            Note that modifying the data used as parameter will not affect the returned array, since it will be a copy if data is not None.
        """
        if (isTensorLike):
            return None
        if (data is None):
            if self.isFloatTensor():
                data = xplib.xp.random.random_sample(shape)
            else:
                data = xplib.xp.random.random_sample(shape) + xplib.xp.random.random_sample(shape) * 1.0j
        elif (data.shape != shape):
            data = xplib.xp.copy(data.reshape(shape))
        else:
            data = xplib.xp.copy(data)
        return data
            
    def deduction(self, legs, shape, labels, data, isTensorLike = False):
        # print('deduction(legs = {}, shape = {}, labels = {}, data = {}, isTensorLike = {})'.format(legs, shape, labels, data, isTensorLike))
        """
        Deduce the legs, shape, labels and data from the input of user. Guess the missing information if not provided.
        For details, check "Notes" of comments for Tensor.

        Parameters
        ----------
        legs : None or list of Leg
            Legs of the tensor that already existed before creating the tensor. If None, then automatically generated.
        shape: None or tuple of int
            Expected shape for the tensor.
        labels : None or list of str
            The labels to be added to the legs of this tensor.
        data : None or ndarray of float
            The data to be put in the tensor, None for randomly generated.
        isTensorLike : bool, default False
            Whether we are working for a TensorLike object: if True, then data is None.

        Returns
        ------- 
        legs : list of Leg

        shape: tuple of int

        labels : list of str
            The labels to be added to the legs of this tensor.
        data : None or ndarray of float
            The data to be put in the tensor. None for isTensorLike = True case.
        
        Notes
        -----
        Although each of the first 4 parameters can be None by default, user must provide enough information for the deduction of the real shape, labels, legs and data(if not TensorLike).
        """
        funcName = "Tensor.deduction"
        if (legs is not None):

            if (not self.checkLegsLabelsCompatible(legs = legs, labels = labels)):
                raise ValueError(funcs.errorMessage('labels {} is not compatible with legs {}'.format(labels, legs), location = funcName))
            if (labels is None):
                labels = [leg.name for leg in legs]

            if (not self.checkLegsShapeCompatible(legs = legs, shape = shape)):
                raise ValueError(funcs.errorMessage('shape {} is not compatible with legs {}'.format(shape, legs), location = funcName))
            if (shape is None):
                shape = tuple([leg.dim for leg in legs]) 

            if (not self.checkShapeDataCompatible(shape = shape, data = data)):
                raise ValueError(funcs.errorMessage('data shape {} is not compatible with required shape {}'.format(data.shape, shape), location = funcName))
        
        elif (shape is not None):

            if (not self.checkShapeLabelsCompatible(shape = shape, labels = labels)):
                raise ValueError(funcs.errorMessage('labels {} is not compatible with required shape {}'.format(labels, shape), location = funcName))
            if (labels is None):
                labels = self.generateLabels(len(shape))
            
            if (not self.checkShapeDataCompatible(shape = shape, data = data)):
                raise ValueError(funcs.errorMessage('data shape {} is not compatible with required shape {}'.format(data.shape, shape), location = funcName))

        elif (data is not None):
            shape = data.shape 
            if (not self.checkShapeLabelsCompatible(shape = shape, labels = labels)):
                raise ValueError(funcs.errorMessage('labels {} is not compatible with required shape {}'.format(labels, shape), location = funcName))
            if (labels is None):
                labels = self.generateLabels(len(shape))

        else:
            raise ValueError(funcs.errorMessage("Tensor() cannot accept parameters where legs, shape and data being None simultaneously.", location = funcName))

        data = self.generateData(shape = shape, data = data, isTensorLike = isTensorLike)
        
        if (legs is None):
            legs = []
            for label, dim in zip(labels, list(shape)):
                legs.append(Leg(self, dim, label))

        else:
            for leg in legs:
                leg.tensor = self

        return legs, shape, labels, data

    @property 
    def labels(self):
        return [leg.name for leg in self.legs]
    
    @property 
    def chi(self):
        return self.shape[0]

    @property
    def dim(self):
        if (self.a is None):
            return len(self.legs)
        else:
            return len(self.a.shape)
    
    @property 
    def shape(self):
        if (self.a is None):
            # print('shape from TensorLike')
            # print(self.legs)
            return tuple([leg.dim for leg in self.legs])
        else:
            return self.a.shape

    @property
    def totalSize(self):
        return funcs.tupleProduct(self.shape)


    def __str__(self):
        if (self.tensorLikeFlag):
            objectStr = 'TensorLike'
        else:
            objectStr = 'Tensor'
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
            objectStr = 'TensorLike'
        else:
            objectStr = 'Tensor'
        if not (self.degreeOfFreedom is None):
            dofStr = ', degree of freedom = {}'.format(self.degreeOfFreedom)
        else:
            dofStr = ''
        if (self.name is not None):
            nameStr = self.name + ', ' 
        else:
            nameStr = ''
        return '{}({}shape = {}, labels = {}{})\n'.format(objectStr, nameStr, self.shape, self.labels, dofStr)

    def __matmul__(self, b):
        return contractTwoTensors(ta = self, tb = b)

    def bondDimension(self):
        """
        Bond dimension of the tensor. Work for the case when all dimensions are the same, otherwise, generate a warning message and return the first dimension.

        Returns
        -------
        int
            The bond dimension of this tensor.
        """

        if (not funcs.checkAllEqual(self.shape)):
            warnings.warn(funcs.warningMessage(warn = "shape of tensor does not contain the same dimesion for all legs: {}".format(self.shape), location = "Tensor.bondDimension"))
        return self.shape[0]

    def generateLabels(self, n):
        """
        Automatically generate labels for legs when labels are not provided. Generated in order of alphabeta.

        Parameters
        ----------
        n : int
            The number of legs to be generated.
        
        Returns
        -------
        list of str
            The list of labels, in the form ['a', 'b', 'c', 'd', ...]
        """

        assert (n <= 26), funcs.errorMessage(err = "Too many dimensions for input shape: {}".format(n), location = "Tensor.generateLabels")
        labelList = 'abcdefghijklmnopqrstuvwxyz'
        return [labelList[i] for i in range(n)]
    
    def indexOfLabel(self, lab, backward = False):
        """
        Index of a given label.
        TODO : warning message for usage of this?

        Parameters
        ----------
        lab : str
            The label of which the index is asked.
        backward : bool, default False
            Whether to search from backward when two or more same labels exist. Used when self inner product is wanted, and obtain the two indices by indexOfLabel(lab) and indexOfLabel(backward = True).

        Returns
        int
            The index of lab. Search from backward if asked.
        """
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

    def indexOfLeg(self, leg):
        """
        Index of a given leg. More stable since each leg is unique.

        Parameters
        ----------
        leg : Leg
            The leg of which the index is asked.
        
        Returns
        -------
        int
            The index of the leg
        """
        return self.legs.index(leg)
    def getLegIndices(self, legs):
        """
        Index of a given set of legs. For details, check the docstring in indexOfLeg.

        Parameters
        ----------
        leg : list of Leg
            The set of legs of which the index is asked.
        
        Returns
        -------
        list of int
            The indices of legs.
        """
        return [self.indexOfLeg(leg) for leg in legs]
    def getLeg(self, label, backward = False):
        """
        Obtain the leg from the label, by finding the index of given label. For more details, check indexOfLabel.

        Parameters
        ----------
        lab : str
            The label of which the index is asked.
        backward : bool, default False
            Whether to search from backward when two or more same labels exist. Used when self inner product is wanted, and obtain the two indices by indexOfLabel(lab) and indexOfLabel(backward = True).

        Returns
        None or Leg
            The leg corresponding to given label. If the given label does not appear, return None.
        """

        index = self.indexOfLabel(label, backward = backward)
        if (index == -1):
            return None
        return self.legs[index]
        # res = None 
        # for leg in self.legs:
        #     if (leg.name == label):
        #         res = leg 
        #         break 
        # assert (res is not None), "Error: {} not in tensor labels {}.".format(label, self.labels)
        
        # return res

    def moveLegsToFront(self, legs):
        """
        Change the orders of legs: move a given set of legs to the front while not modifying the relative order of other legs. Use xplib.xp.moveaxis to modify the data if this is not a TensorLike object.

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
        if (not self.tensorLikeFlag):
            self.a = xplib.xp.moveaxis(self.a, moveFrom, moveTo)

    def toVector(self):
        """
        Flatten the data contained to a 1D-vector.

        Returns
        -------
        1D ndarray of float
            A vector contains the data in this tensor, following the current order of labels.
        
        """

        assert (not self.tensorLikeFlag), funcs.errorMessage('TensorLike cannot be transferred to vector since no data contained.', 'Tensor.toVector')
        return xplib.xp.copy(xplib.xp.ravel(self.a))
    
    def toMatrix(self, rows = None, cols = None):
        """
        Make a matrix of the data of this tensor, given the labels or legs of rows and cols.

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
        # print(rows, cols)
        # print(self.labels)
        # input two set of legs
        assert (not self.tensorLikeFlag), funcs.errorMessage('TensorLike cannot be transferred to matrix since no data contained.', 'Tensor.toMatrix')
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

        data = xplib.xp.moveaxis(xplib.xp.copy(self.a), moveFrom, moveTo)
        data = xplib.xp.reshape(data, (rowTotalSize, colTotalSize))
        return data

    def complementLegs(self, legs):
        """
        Search for the legs that are not in the set of given legs.

        Parameters
        ----------
        legs : list of Leg

        Returns
        -------
        list of Leg
            The leg list after removing given legs.
        """
        return funcs.listDifference(self.legs, legs)

    def copy(self):
        """
        Make a copy of current tensor, without copy the legs. Note that we do not need to pass copy of data, since generateData will always make a copy.

        Legs cannot be connected once: so if we copy the legs, contract the copied tensor and the tensor connected to the original tensor will make the bond invalid. So we do not copy the legs. However, we can make the copy of legs in CTL.tensor.contract.optimalContract.copyTensorList, please check the docs there for more details.

        Returns
        -------
        Tensor
            A copy of the current tensor, all the information can be copied is contained.
        """
        return Tensor(data = self.a, shape = self.shape, degreeOfFreedom = self.degreeOfFreedom, name = self.name, labels = self.labels, diagonalFlag = self.diagonalFlag, tensorLikeFlag = self.tensorLikeFlag)
        # no copy of tensor legs, which may contain connection information
        # the data will be copied in the new tensor, since all data is generated by "generateData"
    def toTensorLike(self):
        """
        Make a copy of current tensor, without copying the legs. This function works almost like self.copy(), but without copying the data.

        Returns
        -------
        Tensor
            A TensorLike of the current tensor, all the information can be copied is contained except legs and data.
        """
        if (self.tensorLikeFlag):
            return self.copy()
        else:
            return Tensor(data = None, degreeOfFreedom = self.degreeOfFreedom, shape = self.shape, name = self.name, labels = self.labels, diagonalFlag = self.diagonalFlag, tensorLikeFlag = True)
    
    def copyN(self, n):
        """
        Make a set of n copies of the current tensor.

        Returns
        -------
        list of Tensor
            A length-n list of copies of current tensor.
        """
        return [self.copy() for _ in range(n)]
    
    def getLabelIndices(self, labs, backward = False):
        """
        Index of a given set of lablels. For details, check the docstring in indexOfLabel. If two same labels have been used, then the first will be searched in given direction and another will be searched in the opposite direction(by default, first from front, the other from backward).

        Parameters
        ----------
        leg : list of str
            The set of labels of which the index is asked.
        backward : bool, default False
            Whether to search from backward.
        
        Returns
        -------
        list of int
            The indices of labels.
        """
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
        """
        Rename a given label to a new name.

        Parameters
        ----------
        changeFrom, changeTo : str

        """
        legIndex = self.indexOfLabel(changeFrom)
        if (legIndex == -1):
            warnings.warn(funcs.warningMessage(warn = 'leg name {} does not exist, no rename happened'.format(changeFrom), location = 'Tensor.renameLabel'), RuntimeWarning)
            return

        self.legs[legIndex].name = changeTo
        # self.legs[changeTo] = self.legs[changeFrom]
        # if (changeFrom != changeTo):
        #     del self.legs[changeFrom]

    def renameLabels(self, changeFrom, changeTo):
        """
        Rename a set of labels to new names.

        Parameters
        ----------
        changeFrom, changeTo : list of str

        """
        assert (len(changeFrom) == len(changeTo)), "Error: renameLabels need two list with equal number of labels, gotten {} and {}".format(changeFrom, changeTo)
        for cf, ct in zip(changeFrom, changeTo):
            self.renameLabel(cf, ct)

    def shapeOfLabel(self, label, backward = False):
        """
        Find the shape of a given label. For how to search the leg of the label, check self.getLeg.

        Parameters
        ----------
        label : str

        backward : bool, default False

        Returns
        -------
        int
            The dimension of the leg of given label. If the label does not exist, return -1.
        """
        leg = self.getLeg(label, backward = backward)
        if (leg is None):
            return -1
        return leg.dim
        # for leg in self.legs:
        #     if leg.name == label:
        #         return leg.dim 
        
        # return -1
    def shapeOfLabels(self, labs, backward = False):
        """
        Find the shape of a given set of labels. For details, check self.getLabelIndices.

        Parameters
        ----------
        labs : list of str

        backward : bool, default False

        Returns
        -------
        int
            The dimension of the legs of given labels.
        """
        return self.shapeOfIndices(self.getLabelIndices(labs, backward = backward))

    def shapeOfIndex(self, index):
        """
        Shape of a given index of legs. This should be used internally, since the order of legs is not fixed.

        Parameters
        ----------
        index : int

        Returns
        -------
        int
            The dimension of leg with index as index.
        """
        return self.shape[index]
    def shapeOfIndices(self, indices):
        """
        Shape of given indices of legs. This should be used internally, since the order of legs is not fixed.

        Parameters
        ----------
        indices : list of int

        Returns
        -------
        tuple of int
            The dimension of legs with indices as indices.
        """
        return tuple([self.shape[x] for x in indices])

    def addTensorTag(self, name):
        """
        Add a tag to all legs of the current tensor, so we can recover the leg structure after contraction of a tensor network.

        For this usage, please not use "-" explicitly in the name of tensors, or it will be considered as a tag. 

        TODO : make a internal / external flag for creating tensors, so it can give a warning if user is creating tensors with labels containing '-'?

        Parameters
        ----------
        name : str
            The tag to be attached to legs.
        
        """
        for leg in self.legs:
            assert (leg.name.find('-') == -1), "Error: leg name {} already has a tensor tag.".format(leg.name)
            leg.name = name + '-' + leg.name 
    
    def removeTensorTag(self):
        """
        Remove the tags of legs. If a leg does not contain a tag, then give a warning message.
        """
        for leg in self.legs:
            divLoc = leg.name.find('-')
            if (divLoc == -1):
                warnings.warn(funcs.warningMessage(warn = "leg name {} of tensor {} does not contain a tensor tag.".format(leg.name, self), location = "Tensor.removeTensorTag"))
            # assert (divLoc != -1), "Error: leg name {} does not contain a tensor tag.".format(leg.name)
            if (divLoc != -1):
                leg.name = leg.name[(divLoc + 1):]    
    
    def getLegsByLabel(self, labelList):
        indices = funcs.generateIndices(self.labels, labelList)
        for index, label in zip(indices, labelList):
            if (index is None):
                raise ValueError(funcs.errorMessage('{} is not in tensor {}'.format(label, self), location = 'Tensor.getLegsByLabel'))
        return [self.legs[index] for index in indices]

    def moveLabelsToFront(self, labelList):
        """
        Change the orders of legs: move a given set of labels to the front. For details, check "self.moveLegsToFront".

        Parameters
        ----------
        labelList : list of str
            The set of labels to be put at front.

        """
        # legs = [self.getLeg(label) for label in labelList]
        legs = self.getLegsByLabel(labelList)
        self.moveLegsToFront(legs)
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
        
        # # print(moveFrom, moveTo)
        # # print(labelList)
        # # print(self.labels)
        # self.legs = movedLegs + self.legs 
        # if (not self.tensorLikeFlag):
        #     self.a = xplib.xp.moveaxis(self.a, moveFrom, moveTo)

    # def outProduct(self, labelList, newLabel):
    #     self.moveLabelsToFront(labelList)
    #     n = len(labelList)
    #     newShape = (-1, ) + self.shape[n:]

    #     if (self.tensorLikeFlag):
    #         newDim = 1
    #         for leg in self.legs[:n]:
    #             newDim *= leg.dim
    #     else:
    #         self.a = xplib.xp.reshape(self.a, newShape)
    #         newDim = self.a.shape[0]
    #     self.legs = [Leg(self, newDim, newLabel)] + self.legs[n:]
    #     # return the new leg, for usage of wider usage
    #     return self.legs[0]

    def outProduct(self, legList, newLabel):
        """
        Do an out product to combine a set of legs to a single leg. 
        Parameters
        ----------
        legList : list of str or list of str
            The legs to be combined, or labels or the legs.
        newLabel : str
            The new label for the combined leg.

        Returns
        -------
        Leg
            The new leg of all legs to be combined.
        """
        assert (isinstance(legList, list) and (len(legList) > 0)), funcs.errorMessage(err = "outProduct cannot work on leg list of zero legs or non-list, {} obtained.".format(legList), location = 'Tensor.outProduct')
        if (isinstance(legList[0], str)):
            return self.outProduct([self.getLeg(label) for label in legList], newLabel)

        # connectedLegs = [leg for leg in legList if (leg.bond is not None)]
        # if (len(connectedLegs) > 0):
        #     warnings.warn(funcs.warningMessage(warn = "out producting legs {} that has been connected: remove the connection.".format(connectedLegs), location = 'Tensor.outProduct'))

        self.moveLegsToFront(legList)
        n = len(legList)
        newShape = (-1, ) + self.shape[n:]

        if (self.tensorLikeFlag):
            newDim = 1
            for leg in self.legs[:n]:
                newDim *= leg.dim
        else:
            self.a = xplib.xp.reshape(self.a, newShape)
            newDim = self.a.shape[0]
        self.legs = [Leg(self, newDim, newLabel)] + self.legs[n:]
        # return the new leg, for usage of wider usage
        return self.legs[0]

    def reArrange(self, labels):
        """
        Rearrange the legs to a given order according to labels.

        Parameters
        ----------
        labels : list of str
            The order of labels after rearranging.
        """
        assert (funcs.compareLists(self.labels, labels)), "Error: tensor labels must be the same with original labels: get {} but {} needed".format(len(labels), len(self.labels))
        self.moveLabelsToFront(labels)

    def norm(self):
        """
        Norm of the current tensor.

        Returns
        -------
        float
            The norm of data.
        """
        assert (not self.tensorLikeFlag), funcs.errorMessage('TensorLike do not have norm since no data contained.', 'Tensor.norm')
        return xplib.xp.linalg.norm(self.a)

    def trace(self, rows = None, cols = None):
        """
        Trace of the current tensor after making a matrix according to rows and cols. For details, check Tensor.toMatrix

        Parameters
        ----------
        rows : None or list of str or list of Leg
            The legs for the rows of the matrix. If None, deducted from cols.
        cols : None or list of str or list of Leg
            The legs for the cols of the matrix. If None, deducted from rows.
        
        Returns
        -------
        float
            The trace of the matrix generated by given cols and rows.
        """
        assert (not self.tensorLikeFlag), funcs.errorMessage('TensorLike do not have trace since no data contained.', 'Tensor.trace')
        mat = self.toMatrix(rows = rows, cols = cols)
        assert (mat.shape[0] == mat.shape[1]), "Error: Tensor.trace must have the same dimension for cols and rows, but shape {} gotten.".format(mat.shape)
        return xplib.xp.trace(mat)

    def single(self):
        """
        Generate a single value from a shapeless tensor.

        Returns
        -------
        float
            A single value of this tensor.

        """
        # return the single value of this tensor
        # only works if shape == (,)
        assert (not self.tensorLikeFlag), funcs.errorMessage('TensorLike cannot be transferred to single value since no data contained.', 'Tensor.single')
        assert self.shape == (), "Error: cannot get single value from tensor whose shape is not ()."
        return self.a

    def toTensor(self, labels = None):
        """
        Return a ndarray of this tensor.

        Parameters
        ----------
        labels : None or list of str
            The order of labels for the output tensor. Note that if labels is None, the order of legs is not fixed, may differ from time to time.
        
        Returns
        -------
        ndarray of float
            The data of the tensor, order of legs are given by the labels.
        """
        assert (not self.tensorLikeFlag), funcs.errorMessage('TensorLike cannot be transferred to tensor since no data contained.', 'Tensor.toTensor')
        if (labels is not None):
            self.reArrange(labels)
        return self.a

    def typeName(self):
        """
        The type of the current class.

        Returns
        -------
        {"Tensor", "TensorLike"}
        """
        if (self.tensorLikeFlag):
            return "TensorLike"
        else:
            return "Tensor"

    def labelInTensor(self, label):
        """
        Whether the given label is one label of the current tensor.
        
        Parameters
        ----------
        label : str

        Returns
        -------
        bool
            Whether the label is of this tensor.
        """
        return label in self.labels 
    def labelsInTensor(self, labels):
        """
        Whether the given labels are all labels of the current tensor.
        
        Parameters
        ----------
        labels : list of str

        Returns
        -------
        bool
            Whether the labels are of this tensor.
        """
        for label in labels:
            if not (label in self.labels):
                return False

        return True

    def sumOutLeg(self, leg, weights = None):
        """
        Sum out one leg to make a (D - 1)-dimensional tensor. Give a warning(and do nothing) if leg is not one of the current tensor, and give a warning if leg is connected to some bond(not free).

        Parameters
        ----------
        leg : Leg
            The leg to be summed out.
        weights : 1-d array, optional
            If not None, then each index on given dimension will be weighted by weights[i].

        """
        if not (leg in self.legs):
            warnings.warn(funcs.warningMessage("leg {} is not in tensor {}, do nothing.".format(leg, self), location = 'Tensor.sumOutLeg'), RuntimeWarning)
            return
        if leg.bond is not None:
            warnings.warn(funcs.warningMessage("leg {} to be summed out is connected to bond {}.".format(leg, leg.bond), location = 'Tensor.sumOutLeg'), RuntimeWarning)
        
        idx = self.legs.index(leg)
        # self.a = xplib.xp.sum(self.a, axis = idx)
        self.a = funcs.sumOnAxis(self.a, axis = idx, weights = weights)
        self.legs = self.legs[:idx] + self.legs[(idx + 1):]
    
    def sumOutLegByLabel(self, label, backward = False, weights = None):
        """
        Sum out one leg to make a (D - 1)-dimensional tensor via the label of leg. Give a warning(and do nothing) if label is not one of the current tensor.

        Parameters
        ----------
        label : str or list of str
            The label of the leg(s) to be summed out.
        backward : bool, default False
            Whether to search from backward of legs.
        weights : 1-d array, optional
            If not None, then each index on given dimension will be weighted by weights[i].

        """
        if isinstance(label, list):
            for l in label:
                self.sumOutLegByLabel(l, backward)
            return 
        leg = self.getLeg(label, backward = backward)
        if leg is None:
            warnings.warn(funcs.warningMessage("leg {} is not in tensor {}, do nothing.".format(label, self), location = 'Tensor.sumOutLegByLabel'), RuntimeWarning)
        self.sumOutLeg(leg, weights = weights)
    
def TensorLike(shape = None, labels = None, data = None, degreeOfFreedom = None, name = None, legs = None, diagonalFlag = False):
    """
    Make a TensorLike, in the form like creating a tensor. For the details, check __init__ of Tensor
    """
    
    return Tensor(shape = shape, labels = labels, data = data, degreeOfFreedom = degreeOfFreedom, name = name, legs = legs, diagonalFlag = diagonalFlag, tensorLikeFlag = True)

from CTL.tensor.contract.contract import contractTwoTensors