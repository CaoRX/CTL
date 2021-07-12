# # what we want here: 
# # an object that behaves almost the same as "Tensor(also DiagonalTensor)"
# # but at the same time, without data inside

# import CTL.funcs.funcs as funcs

# class TensorLike:

#     # attributes: xp, totalSize, labels, contractLabels, degreeOfFreedom

#     # bondNameSet = set([])

#     # how to implement diagonal tensor?
#     # we want: 1. used just as the Tensor, and 
#     # 2. with lower cost in contraction
#     # solution 1. another DiagonalTensor class, implementing all existing functions (hard to append new functions)
#     # solution 2. a class inherit Tensor, and rewrite the methods if needed

#     def __init__(self, shape = None, labels = None, degreeOfFreedom = None, name = None, legs = None, diagonalFlag = False):

#         self.tensorLikeFlag = True 
#         self.diagonalFlag = diagonalFlag 
#         self.name = name
#         self.degreeOfFreedom = degreeOfFreedom

#         # deduce the shape of tensor via (legs, shape, labels)
#         assert (not ((data is None) and (shape is None))), "Error: TensorBase must be initialized with either data or shape."
#         if (shape is None):
#             shape = data.shape
#         # print(shape, data.shape)

#         assert ((labels is None) or (len(shape) == len(labels))), "Error: the number of labels input is {}, while the dimension is {}.".format(len(labels), len(shape))

#         self.xp = np
#         # self.shape = shape
#         self.totalSize = funcs.tupleProduct(shape)
#         if (data is None):
#             #self.a = self.xp.zeros(self.shape, dtype = self.xp.float64)
#             self.a = self.xp.random.random_sample(shape)
#         else:
#             self.a = self.xp.copy(data)
#         assert (self.totalSize == funcs.tupleProduct(self.a.shape)), 'Error: expect {} elements but {} gotten.'.format(self.totalSize, funcs.tupleProduct(self.a.shape))
#         if (self.a.shape != shape):
#             self.a = self.xp.reshape(self.a, shape)
#         # print('a shape = {}'.format(self.a.shape))
    
#         if (labels is None):
#             labels = self.generateLabels(len(shape))
#         self.degreeOfFreedom = degreeOfFreedom
#         self.name = name

#         if (legs is None):
#             self.legs = []
#             for label, dim in zip(labels, list(self.shape)):
#                 self.legs.append(Leg(self, dim, label))
#         else:
#             assert (len(legs) == self.dim), "Error: number of legs and dim are not compatible in Tensor.__init__(): {} and {}.".format(len(legs), self.dim)
#             self.legs = legs 
#             for leg in self.legs:
#                 leg.tensor = self

#     @property 
#     def labels(self):
#         return [leg.name for leg in self.legs]
    
#     @property 
#     def chi(self):
#         return self.shape[0]


#     def __str__(self):
#         if not (self.degreeOfFreedom is None):
#             dofStr = ', degree of freedom = {}'.format(self.degreeOfFreedom)
#         else:
#             dofStr = ''
#         if (self.name is not None):
#             nameStr = self.name + ', ' 
#         else:
#             nameStr = ''
#         return 'Tensor({}shape = {}, labels = {}{})'.format(nameStr, self.shape, self.labels, dofStr)

#     def __repr__(self):
#         if not (self.degreeOfFreedom is None):
#             dofStr = ', degree of freedom = {}'.format(self.degreeOfFreedom)
#         else:
#             dofStr = ''
#         if (self.name is not None):
#             nameStr = self.name + ', ' 
#         else:
#             nameStr = ''
#         return 'Tensor({}shape = {}, labels = {}{})\n'.format(nameStr, self.shape, self.labels, dofStr)

#     def bondDimension(self):
#         return self.shape[0]

#     def generateLabels(self, n):
#         assert (n <= 26), "Too many dimensions for input shape"
#         labelList = 'abcdefghijklmnopqrstuvwxyz'
#         return [labelList[i] for i in range(n)]
    
#     def indexOfLabel(self, lab, backward = False):
#         labels = self.labels
#         if not (lab in labels):
#             return -1
#         if (backward): # search from backward, for a list can have more than one same label
#             labels.reverse()
#             ret = len(labels) - labels.index(lab) - 1
#             labels.reverse()
#         else:
#             ret = labels.index(lab)
#         return ret

#     def getLegIndex(self, leg):
#         return self.legs.index(leg)
#     def getLegIndices(self, legs):
#         return [self.getLegIndex(leg) for leg in legs]
#     def getLeg(self, label):
#         res = None 
#         for leg in self.legs:
#             if (leg.name == label):
#                 res = leg 
#                 break 
#         assert (res is not None), "Error: {} not in tensor labels {}.".format(label, self.labels)
        
#         return res

#     def moveLegsToFront(self, legs):
#         moveFrom = []
#         moveTo = []
#         currIdx = 0
#         movedLegs = legs
#         for currLeg in legs:
#             for i, leg in enumerate(self.legs):
#                 if (leg == currLeg):
#                     moveFrom.append(i)
#                     moveTo.append(currIdx)
#                     currIdx += 1
#                     break

#         for leg in movedLegs:
#             self.legs.remove(leg)
        
#         # print(moveFrom, moveTo)
#         # print(labelList)
#         # print(self.labels)
#         self.legs = movedLegs + self.legs 
#         self.a = self.xp.moveaxis(self.a, moveFrom, moveTo)

#     def toVector(self):
#         return self.xp.copy(self.xp.ravel(self.a))
    
#     def toMatrix(self, rows, cols):
#         # print(rows, cols)
#         # print(self.labels)
#         # input two set of legs
#         assert not ((rows is None) and (cols is None)), "Error in Tensor.toMatrix: toMatrix must have at least row or col exist."
#         if (rows is not None) and (isinstance(rows[0], str)):
#             rows = [self.getLeg(label) for label in rows]
#         if (cols is not None) and (isinstance(cols[0], str)):
#             cols = [self.getLeg(label) for label in cols]
#         if (cols is None):
#             cols = funcs.listDifference(self.legs, rows)
#         if (rows is None):
#             rows = funcs.listDifference(self.legs, cols)
#         assert (funcs.compareLists(rows + cols, self.legs)), "Error Tensor.toMatrix: rows + cols must contain(and only contain) all legs of tensor."

#         colIndices = self.getLegIndices(cols)
#         rowIndices = self.getLegIndices(rows)

#         colShape = tuple([self.shape[x] for x in colIndices])
#         rowShape = tuple([self.shape[x] for x in rowIndices])
#         colTotalSize = funcs.tupleProduct(colShape)
#         rowTotalSize = funcs.tupleProduct(rowShape)

#         moveFrom = rowIndices + colIndices
#         moveTo = list(range(len(moveFrom)))

#         data = self.xp.moveaxis(self.xp.copy(self.a), moveFrom, moveTo)
#         data = self.xp.reshape(data, (rowTotalSize, colTotalSize))
#         return data

#     def complementLegs(self, legs):
#         return funcs.listDifference(self.legs, legs)

#     def copy(self):
#         return Tensor(data = self.xp.copy(self.a), degreeOfFreedom = self.degreeOfFreedom, name = self.name, labels = self.labels)
#         # no copy of tensor legs, which may contain connection information
    
#     def copyN(self, n):
#         return [self.copy() for _ in range(n)]
    
#     def getLabelIndices(self, labs, backward = False):
#         ret = []
#         for i, lab in enumerate(labs):
#             if (labs[:i].count(lab) > 0):
#                 ret.append(self.indexOfLabel(lab, backward = (not backward)))
#                 # if there are two same labels in labs
#                 # then the first take from front, the second from back
#             else:
#                 ret.append(self.indexOfLabel(lab, backward = backward))

#         return ret

#     def renameLabel(self, changeFrom, changeTo):
#         self.legs[self.indexOfLabel(changeFrom)].name = changeTo
#         # self.legs[changeTo] = self.legs[changeFrom]
#         # if (changeFrom != changeTo):
#         #     del self.legs[changeFrom]

#     def renameLabels(self, changefrom, changeto):
#         assert (len(changefrom) == len(changeto)), "Error: renameLabels need two list with equal number of labels, gotten {} and {}".format(changefrom, changeto)
#         for cf, ct in zip(changefrom, changeto):
#             self.renameLabel(cf, ct)

#     def shapeOfLabel(self, label):
#         for leg in self.legs:
#             if leg.name == label:
#                 return leg.dim 
        
#         return -1
#     def shapeOfLabels(self, labs):
#         return self.shapeOfIndices(self.getLabelIndices(labs))

#     def shapeOfIndex(self, index):
#         return self.shape[index]
#     def shapeOfIndices(self, indices):
#         return tuple([self.shape[x] for x in indices])

#     def addTensorTag(self, name):
#         for leg in self.legs:
#             assert (leg.name.find('-') == -1), "Error: leg name {} already has a tensor tag.".format(leg.name)
#             leg.name = name + '-' + leg.name 
    
#     def removeTensorTag(self):
#         for leg in self.legs:
#             divLoc = leg.name.find('-')
#             assert (divLoc != -1), "Error: leg name {} does not contain a tensor tag.".format(leg.name)
#             leg.name = leg.name[(divLoc + 1):]

#     def moveLabelsToFront(self, labelList):
#         moveFrom = []
#         moveTo = []
#         currIdx = 0
#         movedLegs = []
#         for label in labelList:
#             for i, leg in enumerate(self.legs):
#                 if (leg.name == label):
#                     moveFrom.append(i)
#                     moveTo.append(currIdx)
#                     currIdx += 1
#                     movedLegs.append(leg)
#                     break

#         for leg in movedLegs:
#             self.legs.remove(leg)
        
#         # print(moveFrom, moveTo)
#         # print(labelList)
#         # print(self.labels)
#         self.legs = movedLegs + self.legs 
#         self.a = self.xp.moveaxis(self.a, moveFrom, moveTo)

#     def outProduct(self, labelList, newLabel):
#         self.moveLabelsToFront(labelList)
#         n = len(labelList)
#         newShape = (-1, ) + self.shape[n:]
#         self.a = np.reshape(self.a, newShape)
#         # self.shape = self.a.shape

#         newDim = self.a.shape[0]
#         self.legs = [Leg(self, newDim, newLabel)] + self.legs[n:]

#     def reArrange(self, labels):
#         assert (funcs.compareLists(self.labels, labels)), "Error: tensor labels must be the same with original labels: get {} but {} needed".format(len(labels), len(self.labels))
#         self.moveLabelsToFront(labels)

#     def norm(self):
#         return self.xp.linalg.norm(self.a)

#     def trace(self, rows = None, cols = None):
#         mat = self.toMatrix(rows = rows, cols = cols)
#         assert (mat.shape[0] == mat.shape[1]), "Error: Tensor.trace must have the same dimension for cols and rows, but shape {} gotten.".format(mat.shape)
#         return self.xp.trace(mat)

#     def single(self):
#         # return the single value of this tensor
#         # only works if shape == (,)
#         assert self.shape == (), "Error: cannot get single value from tensor whose shape is not ()."
#         return self.a

#     def toTensor(self, labels = None):
#         if (labels is not None):
#             self.reArrange(labels)
#         return self.a