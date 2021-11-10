from collections import Counter
# import numpy as np 
import CTL.funcs.xplib as xplib
import string
import random
import math
import warnings
from copy import deepcopy

def deprecatedFuncWarning(funcName, fileName = None, newFuncName = None, deprecateMessage = None):
    """
    Warn if a function has been deprecated.

    Parameters
    ----------
    funcName : str
        The name of function that has been deprecated.
    fileName : str, optional
        The name of the file where the deprecated function locates, for understanding the details.
    newFuncName : str, optional
        The name of new functions work for a similar task, help the user to change the usage of deprecated function to a new one.
    deprecateMessage : str, optional
        Some extra texts for explaining why this function has been deprecated, or suggestions for other choices.
    """
    if (fileName is None):
        fileNameInfo = ''
    else:
        fileNameInfo = 'in {} '.format(fileName)
    if (newFuncName is None):
        if (deprecateMessage is None):
            deprecateMessage = "This function should not be used anywhere."
        warnings.warn('Warning: {} {}has been deprecated. {}'.format(funcName, fileNameInfo, deprecateMessage))
    else:
        warnings.warn('Warning: {} {}has been deprecated. Please use {} instead.'.format(funcName, fileNameInfo, newFuncName))

def listDifference(total, used):
    """
    Return the remaining part of "total" after removing elements in "used". Each element will be removed only once.

    Parameters
    ----------
    total : list of any
        The original list from which we want to remove some elements.
    used : list of any
        The elements we want to remove from total.

    Returns
    -------
    list of any
        The list total after removing elements in used.
    """
    ret = list(total)
    for x in used:
        ret.remove(x)
    return ret

def deleteAllInUsed(total, used):
    """
    Return the list "total" after removing all appearances of elements in "used".

    Parameters
    ----------
    total : list-like of any
        The original list from which we want to remove some elements.
    used : iterable of any
        The elements we want to remove from total.

    Returns
    -------
    list of any
        The list total after removing all elements appearing in used.
    """
    usedSet = set(used)
    ret = list(total)
    return [x for x in ret if not (x in usedSet)]

def tupleRemoveByIndex(initTuple, indexList):
    """
    Remove the elements of given indices from a tuple.

    Parameters
    ----------
    initTuple : tuple of any
        The given tuple from which we will remove elements.
    indexList : list of ints
        The indices of elements that we want to remove.

    Returns 
    -------
    tuple of any
        The tuple after removing elements from initTuple
    """
    initList = list(initTuple)
    indexSet = set(indexList)
    resList = []
    for i in range(len(initList)):
        if not (i in indexSet):
            resList.append(initList[i])
    return tuple(resList)

def tupleProduct(shapeTuple):
    """
    Calculate the product of elements in a tuple. Used for calculating total shape.
    
    Parameters
    ----------
    shapeTuple : iterable of int or float
        The shape-like object we are asking for total shape(product).
    
    Returns
    -------
    int/float
        The product of elements in shapeTuple.
    """
    res = 1
    for x in shapeTuple:
        res *= x
    return res

def spinChainProductSum(spins):
    """
    Calculate the Ising nearest neighbor interactions of a spin chain, periodic boundary condition(PBC).

    Parameters
    ----------
    spins : list of ints or floats
        The given spin under PBC.

    Returns
    float
        The nearest neighbor interactions(products).
    """
    res = 0.0
    n = len(spins)
    for i in range(n):
        res += spins[i] * spins[i + 1 - n]
    return res

def compareLists(a, b):
    """
    Compare two lists, return true if they are just the same under a permutation.

    Parameters
    ----------
    a, b : list of any
        The two lists to compare.
    
    Returns
    bool
        whether a and b are the same under a permutation.
    """
    return Counter(a) == Counter(b)

def commonElements(x, y):
    """
    Find the common elements in two lists.
    Parameters
    ----------
    x, y : list of any
        The two lists of which we want the common elements.
    
    Returns
    -------
    list of any
        The list of common elements. The same elements will only appear once.
    """
    xc = Counter(x)
    yc = Counter(y)
    return list((xc & yc).elements())

def listSymmetricDifference(x, y):
    """
    Symmetric Difference of two lists.

    Parameters
    ----------
    x, y: list of any
        The two list of which we want the symmetric difference.

    Returns
    -------
    list of any
        The list of elements that appear exactly once in either x or y.
    """
    return list(set(x) ^ set(y))

def intToBitList(x, dim):
    """
    Transfer a bit-mask to a fixed-length list of bits.

    Parameters
    ----------
    x : int
        A positive integer as a bit-mask.
    dim : int
        The length of bit list as the output.

    Returns
    -------
    list of ints
        0 / 1's of the bit-mask, of size (dim, )
    """
    res = []
    for _ in range(dim):
        res.append(x % 2)
        x = x // 2

    return res 

def intToBitTuple(x, dim):
    """
    Transfer a bit-mask to a fixed-length tuple of bits.

    Parameters
    ----------
    x : int
        A positive integer as a bit-mask.
    dim : int
        The length of bit list as the output.

    Returns
    -------
    tuple of ints
        0 / 1's of the bit-mask, of size (dim, )
    """
    return tuple(intToBitList(x, dim))

def safeMod(x, m):
    """
    The safe module of integers, work for both positive and negative integers.

    Parameters
    ----------
    x : int
        An integer to take module.
    m : int
        A positive integer as the module.

    Returns
    -------
    int
        A positve integer in [0, m) as the remainder of x.
    """
    return ((x % m) + m) % m

def intToDBaseTuple(x, dim, D):
    """
    Transfer a integer to a base-D fixed-length tuple of digits.

    Parameters
    ----------
    x : int
        A positive integer to be transformed into base-D.
    dim : int
        The length of digit tuple as the output.
    D : int
        The base of the digit tuple.

    Returns
    -------
    tuple of ints
        [0, D - 1) of the digit tuple, of size (dim, )
    """
    res = []
    for _ in range(dim):
        res.append(x % D)
        x = x // D
    return tuple(res)

def triangleWeight(sm, J):
    """
    (Deprecated)Calculate the weight of a triangle of Ising spins, depending on the number of same spin pairs.

    Parameters
    ----------
    sm : int
        The number of spin pairs that are the same on the triangle.
    J : float
        The Ising interaction parameter of H = -J sum s_i s_{i + 1}.
    
    Returns
    -------
    float
        If the number sm is invalid(an odd number), return 0.0; otherwise return the Ising Boltzmann weight.
    """
    # print('Warning: triangle_weight in funcs/funcs.py has been deprecated. This function should not be used anywhere.')
    deprecatedFuncWarning(funcName = 'triangleWeight', fileName = 'funcs/funcs')
    if (sm == 0):
        return 1
    if (sm == 2):
        return xplib.xp.exp(J)
    return 0.0

def loadData(fileName):
    """
    Load the data of a text file containing a float array.

    Parameters
    ----------
    fileName : str
        The name of file we want to load.
    
    Returns
    -------
    numpy.ndarray
        A 2-d array of the file, columns and rows as two dimensions.
    """
    f = open(fileName, 'r')
    data = xplib.xp.array([xplib.xp.array([xplib.xp.float(x) for x in line.strip().split(' ')]) for line in f])
    return data

def randomString(n = 10):
    """
    Generate a random string of form [a-zA-Z]{n}(ascii letters of length n).

    Parameters
    ----------
    n : int, default 10
        The length of the random string.
    
    Returns
    -------
    str
        A random string consisting ascii letters of length n.
    """
    return ''.join(random.choice(string.ascii_letters) for i in range(n))

def identityError(a):
    """
    Calculate the distance of matrix a to a 2D eye matrix

    Parameters
    ----------
    a : 2-D ndarray.
    
    Returns
    -------
    float
        The norm of matrix (a - I).
    """
    return xplib.xp.linalg.norm(a - xplib.xp.eye(a.shape[0], a.shape[1]))

def diagError(a):
    """
    Calculate the error of matrix a to a 2D diagonal matrix
    
    Parameters
    ----------
    a : 2-D ndarray.
    
    Returns
    -------
    float
        The norm of matrix (a - a.diag).
    """
    return xplib.xp.linalg.norm(a - xplib.xp.diag(xplib.xp.diagonal(a))) / xplib.xp.linalg.norm(a)

def checkIdentity(a, eps = 1e-10):
    """
    Calculate the distance of matrix a to a 2D eye matrix

    Parameters
    ----------
    a : 2-D ndarray.

    eps : float, optional. 
        The threshold norm of difference between which we will consider two matrices as the same. By default, 1e-10.
    
    Returns
    -------
    bool
        Whether a is an identity matrix.
    """
    return identityError(a) < eps

def symmetricError(a):
    """
    Calculate the distance of matrix a to a symmetric matrix.

    Parameters
    ----------
    a : 2-D ndarray.
    
    Returns
    -------
    float
        Norm(a - a.T) / Norm(a), a measure of how symmetric a is.
    """
    return xplib.xp.linalg.norm(a - xplib.xp.transpose(a)) / xplib.xp.linalg.norm(a)

def transposeConjugate(a):
    """
    Calculate the transpose conjugate of a matrix.

    Parameters
    ----------
    a : 2-D ndarray, shape of (h, w)
    
    Returns
    -------
    2-D ndarray, shape of (w, h)
        The transpose conjugate of matrix a.
    """
    return xplib.xp.conjugate(xplib.xp.transpose(a))

def aDaggerAProduct(a):
    """
    Calculate the product of transpose conjugate of a matrix and the matrix itself.

    Parameters
    ----------
    a : 2-D ndarray, shape of (h, w)
    
    Returns
    -------
    2-D ndarray, shape of (w, w)
        a^{dagger} @ a
    """
    return xplib.xp.matmul(transposeConjugate(a), a)

def aADaggerProduct(a):
    """
    Calculate the product of a matrix and its transpose conjugate.

    Parameters
    ----------
    a : 2-D ndarray, shape of (h, w)
    
    Returns
    -------
    2-D ndarray, shape of (h, h)
        a @ a^{dagger}
    """
    return xplib.xp.matmul(a, transposeConjugate(a))

def projectorError(a):
    """
    Calculate how much a is like a projector.

    Parameters
    ----------
    a : 2-D ndarray, shape of (h, w).
    
    Returns
    -------
    float
        The distance of a^{dagger} @ a to identity.
    """
    return identityError(aDaggerAProduct(a))

def getDiffError(a, b):
    """
    (Deprecated)Calculate the distance square between a and b.

    Parameters
    ----------
    a, b : ndarray of float, in the same shape
        The two matrices to be compared
    
    Returns
    -------
    float
        The sum of element-wise distance square between a and b.
    """
    deprecatedFuncWarning(funcName = 'getDiffError', fileName = 'CTL.funcs.funcs')
    return xplib.xp.linalg.norm(a ** 2) + xplib.xp.linalg.norm(b ** 2) - xplib.xp.linalg.norm(a * b) * 2

def matDiffError(a, b):
    """
    Calculate the relative distance between a and b.

    Parameters
    ----------
    a, b : ndarray of float, in the same shape
        The two matrices to be compared
    
    Returns
    -------
    float
        Norm(a - b) / Norm(a)
    """
    return xplib.xp.linalg.norm(a - b) / xplib.xp.linalg.norm(a)

def randomArray(shape):
    """
    Generate a random array with given shape, each element is uniformly chosen in [-1, 1)

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of required random array.
    
    Returns
    -------
    ndarray of floats, shape as given shape
        A random array uniformly in [-1, 1)
    """
    return 2.0 * xplib.xp.random.random(shape) - 1.0

# def print_array(a, eps = 1e-15):
# 	deprecatedFuncWarning(funcName = 'print_array', fileName = 'funcs/funcs')
# 	a_copy = a.copy()
# 	a_non_zero = a_copy[xplib.xp.abs(a_copy) > eps]
# 	return a_non_zero

def assertInSet(x, xSet, name):
    """
    Make an assertion that an element is in some set, if not, generating standard error message.
    Parameters
    ----------
    x : any
        The element that should be in some set.
    xSet : set of any
        The set that should contain x.
    name : str
        The name of the element, used to generate error message
    """
    assert (x in xSet), 'Error: {} should be one of {}, but {} gotten.'.format(name, xSet, x)

# def dealParams(paramsDict, paramsNeed, paramsPassed):
# 	"""

# 	"""
# 	for key in paramsPassed:
# 		if (key in paramsDict):
# 			paramsDict[key] = paramsPassed[key]

# 	for key in paramsNeed:
# 		if (paramsDict[key] is None):
# 			return False, key

# 	return True, None

# def paramsFuncMaker(paramsDict, paramsNeed, initialFuncs, initialFuncsName):

# 	paramsDictLocal = deepcopy(paramsDict)
# 	paramsNeedLocal = deepcopy(paramsNeed)

# 	def paramsFunc(paramsPassed):
# 		# print('paramsDict = {}'.format(paramsDictLocal))
# 		paramsDictCopy = deepcopy(paramsDictLocal)
# 		paramsNeedCopy = deepcopy(paramsNeedLocal)
# 		paramsOk, errorMessage = dealParams(paramsDict = paramsDictCopy, paramsNeed = paramsNeedCopy, paramsPassed = paramsPassed)

# 		assert (paramsOk), 'Error: parameter {} is needed in function {} but not passed.\ndefault parameters = {}\nparameters needed = {}.'.format(errorMessage, initialFuncsName, paramsDictLocal, paramsNeedLocal)
# 		return initialFuncs(**paramsDictCopy)

# 	return paramsFunc

# def paramsFuncPackager(paramsDict, paramsNeed, initialFuncs, initialFuncsName, paramsName = 'params'):

# 	# work for methods with parameters, and a parameter called params
# 	# initialFuncs needs all params in the paramsDict as params
# 	# however, we want to set a "default" params, that should be done by this packager

# 	# return a function which takes just the same parameters as initialFuncs, including a "params"
# 	# however, when taking small set of params, we should still generate the whole paramsDict with default values here

# 	paramsDictLocal = deepcopy(paramsDict)
# 	paramsNeedLocal = deepcopy(paramsNeed)

# 	def paramsFunc(**kwargs):
# 		# print('paramsDict = {}'.format(paramsDictLocal))
# 		paramsDictCopy = deepcopy(paramsDictLocal)
# 		paramsNeedCopy = deepcopy(paramsNeedLocal)
# 		if (paramsName in kwargs) and (not (kwargs[paramsName] is None)):
# 			paramsOk, errorMessage = dealParams(paramsDict = paramsDictCopy, paramsNeed = paramsNeedCopy, paramsPassed = kwargs[paramsName])
# 			assert (paramsOk), 'Error: parameter {} is needed in function {} but not passed.\ndefault parameters = {}\nparameters needed = {}.'.format(errorMessage, initialFuncsName, paramsDictLocal, paramsNeedLocal)

# 		kwargs[paramsName] = paramsDictCopy
# 		return initialFuncs(**kwargs)

# 	return paramsFunc

def getIndexDict(idx, labels, shape):
    """
    Get the index tuple of an index in given shape, with the labels.

    Parameters
    ----------
    idx : int
        The index that will be decomposed.
    labels : list of str
        The labels corresponding to each dimension.
    shape : tuple of int
        The shape of given object.

    Returns
    -------
    dict of (str, int)
        The index for each dimension, labelled by the corresponding labels.
    """
    res = dict([])
    for i in range(len(shape) - 1, -1, -1):
        res[labels[i]] = idx % shape[i]
        idx //= shape[i]
    return res

def getNewString(knownSet, n = 10):
    """
    Generate a random string that does not appear in a given set.

    Parameters
    ----------
    knownSet : set of str
        The set that the generated string should not be in.
    n : int, optional.
        The length of the string to be generated. By default, 10.

    Returns
    -------
    str
        A random string of length n, does not appear in knownSet.

    """
    res = randomString(n = n)
    while (res in knownSet):
        res = randomString(n = n)
    return res

def divideIntoKParts(n, k):
    """
    An iterator of the divisions of n to k parts, each element can be in [0, n].

    Parameters
    ----------
    n : int
        The number to be divided.
    k: int
        The number of parts that n should be divided into.

    Yields
    -------
    list of int
        All different lists, each of length k, and the sum is n.
    """
    if (k == 1):
        yield [n]
        return 

    for i in range(n + 1):
        for partAfter in divideIntoKParts(n - i, k - 1):
            yield [i] + partAfter
    return

def myComb(n, p):
    """
    C(n, p), polyfill for python < 3.7. For higher python version, math.comb should be better.

    Parameters
    ----------
    n, p : int

    Returns
    -------
    int
        C(n, p)

    """
    res = 1
    for i in range(p):
        res *= (n - i)
    for i in range(p):
        res = res // (i + 1)
    return res

def calculateDivisionTimes(division):
    """
    For a division of n, calculate how many ways n different elements can be divided into this division.

    Parameters
    ----------
    division : list of int
    
    Returns
    -------
    int
        The number of ways that we can divide sum(division) elements into such a division.
    """
    res = 1
    n = sum(division)
    for p in division:
        res *= myComb(n, p)
        n -= p
    return res

def getValidValues(values, threshold = 1e10):
    """
    Get the valid values in a list. Valid means the absolute value is lower than some threshold. For numpy arrays, consider values[np.abs(values) < threshold]

    Parameters
    ----------
    values : list of floats

    threshold : float, optional.
        The absolute value threshold of values. By default, 1e10.
    
    Returns
    -------
    resX : list of int
        The indices of valid elements.
    resY : list of floats
        The values of valid elements.
    """
    resX = []
    resY = []
    for i, value in enumerate(values):
        if (abs(value) < threshold):
            resX.append(i)
            resY.append(value)
    return resX, resY

def normalizeInfinity(a):
    """
    Normalize array a so that the maximum absolute value is 1.

    Parameters
    ----------
    a : ndarray of float
        The array to be normalized.
    
    Returns
    -------
    ndarray of float, same shape as a
        The normalized array.
    """
    return a / xplib.xp.max(xplib.xp.abs(a))

def binarySearch(left, right, largeEnough, eps):
    """
    Binary search on a real number region.

    Parameters
    ----------
    left, right : float
        The range [left, right] that will be searched. It is recommended that make sure the dividing point is between [left, right], or the result may not be the answer wanted.
    largeEnough : callable
        A callable(function) that accepts one float number, and decide whether it is large enough or not for the search. It is required that it should be monotonic: over some number it should be True, while below the number it returns False.
    eps : float
        The required accuracy of the binary search.

    Returns
    -------
    float
        The value just over which can make largeEnough return True, and below which it returns False.
    """
    assert (left < right), "binary search must start from where (left < right), but get left = {} and right = {}.".format(left, right)
    l = left 
    r = right
    mid = (l + r) * 0.5
    while (r - l > eps):
        if (largeEnough(mid)):
            r = mid 
        else:
            l = mid 
        mid = 0.5 * (l + r)
    return mid

def getCombineSet(listDict, names):
    """
    Calculate the union of a dict of lists.

    Parameters
    ----------
    listDict : dict of list of any
        The dict of lists to be combined.
    names : list of str
        The keys of lists to be combined.

    Returns
    -------
    list of any
        The union of lists in listDict whose key is in names.
    """
    res = []
    for name in names:
        assert (name in listDict), 'Error: name asked {} does not exist in listDict with keys {}.'.format(name, list(listDict.keys()))
        res = res + listDict[name]
    return set(res)

def getCutoff(s, threshold):
    """
    Calculate the first index in a sorted list that is lower than given threshold.

    Parameters
    ----------
    s : list of float
        Sorted list.
    threshold : float
    
    Returns 
    -------
    int
        The first index that s[i] < threshold. If s[i] > threshold for all i, return len(s).
    """
    for i in range(len(s)):
        if (s[i] < threshold):
            return i
    return len(s)

def partialInverse(s, cutoff):
    """
    Calculate the partial inverse of an array.

    Parameters
    ----------
    s: list of float

    cutoff : float
        Value below the cutoff will be considered as zero and not inversed.

    Returns
    -------
    list of float
        1.0 / s for all non-zero elements, and 0 for zero(< threshold) elements.
    """
    res = xplib.xp.zeros(len(s))
    for i in range(cutoff):
        res[i] = 1.0 / s[i]
    return res

def oppositeDirection(x):
    """
    Get the opposite direction of given string.

    Parameters
    ----------
    x : str
        Consisting of 'l', 'r', 'u', 'd'.

    Returns 
    -------
    str
        Consisting of 'l', 'r', 'u', 'd'. Char-wise opposite of x.
    """
    res = ""
    for i in range(len(x)):
        res = res + oppositeSingleDirection(x[i])
    return res

def oppositeSingleDirection(x):
    """
    Get the opposite direction of given direction.

    Parameters
    ----------
    x : str
        One of 'l', 'r', 'u', 'd'.

    Returns 
    -------
    str
        One of 'l', 'r', 'u', 'd'. Opposite of x.
    """
    assertInSet(x, ['l', 'r', 'u', 'd'], 'direction')
    if (x == 'l'):
        return 'r'
    if (x == 'r'):
        return 'l'
    if (x == 'u'):
        return 'd'
    if (x == 'd'):
        return 'u'

def getInRangeFunc(low, high):
    """
    Return a callable that decides whether a float number is in (low, high)

    Parameters 
    ----------
    low, high : float
    
    Returns
    -------
    callable :: float -> bool
        When apply res to a float number, return whether the number is in (low, high) or not.
    """
    return lambda x: (x > low and x < high)

def mapToDict(dictShape, x):
    """
    Make a dict over two lists.

    Parameters
    ----------
    dictShape : list of any
        The labels of the returned dict.
    x : list of any
        The values of the returned dict.

    Returns
    -------
    dict
        Each key in dictShape corresponds to the value in x.
    """
    res = dict([])
    for key in dictShape:
        res[key] = x
    return res

# class LinearRegressionModel:

# 	def __init__(self, x, y):
# 		self.fit(x, y)

# 	def fit(self, x, y):
# 		self.x = xplib.xp.array(deepcopy(x))
# 		self.y = xplib.xp.array(deepcopy(y))

# 		averageX = xplib.xp.average(self.x)
# 		averageY = xplib.xp.average(self.y)
# 		averageXX = xplib.xp.average(self.x ** 2)
# 		averageYY = xplib.xp.average(self.y ** 2)
# 		averageXY = xplib.xp.average(self.x * self.y)

# 		self.k = (averageXY - averageX * averageY) / (averageXX - averageX ** 2)
# 		self.b = averageY - self.k * averageX 
# 		self.r = (averageXY - averageX * averageY) / xplib.xp.sqrt((averageXX - averageX ** 2) * (averageYY - averageY ** 2))

# 	def predict(self, x):
# 		return self.k * x + self.b

def tupleSwap(tp):
    """
    Swap two elements of a tuple.

    Parameters
    ----------
    tp : tuple of any
        Denoted by (a, b).

    Returns
    -------
    tuple of any
        (b, a).

    """
    a, b = tp
    return (b, a)

def floatEqual(a, b, eps = 1e-7):
    """
    Decide whether two float numbers are equal up to a small distance.

    Parameters
    ----------
    a, b : float
        The two numbers to be compared.
    eps : float, default 1e-7
        The maximum error below which we can consider two numbers as equal.

    Returns
    -------
    bool
        Whether the two numbers are equal.
    """
    return abs(a - b) < eps

def floatRelativeEqual(a, b, eps = 1e-7):
    """
    Decide whether two float numbers are equal up to a small distance, relatively.

    Parameters
    ----------
    a, b : float
        The two numbers to be compared.
    eps : float, default 1e-7
        The maximum relative error below which we can consider two numbers as equal.

    Returns
    -------
    bool
        Whether the two numbers are equal.
    """
    return abs(a - b) < eps * 0.5 * (a + b)

def floatArrayEqual(a, b, eps = 1e-7):
    """
    Decide whether two float arrays are equal up to a small distance.

    Parameters
    ----------
    a, b : ndarray
        The two arrays to be compared.
    eps : float, default 1e-7
        The maximum error below which we can consider two numbers as equal.

    Returns
    -------
    bool
        Whether the two arrays are equal.
    """
    if (a.shape != b.shape):
        return False
    return floatEqual(xplib.xp.ravel(a), xplib.xp.ravel(b), eps).all()

def identicalTensorDict(tensor, names):
    """
    Crete a dict, keys of which are names, and values are all given element(tensor).

    Parameters
    ----------
    tensor : any
        The element to be the value for all keys.
    names : list of any
        The keys of returned dict.

    Returns
    -------
    dict
        The dict that has names as keys, and tensor as values for all keys.
    """
    res = dict()
    for name in names:
        res[name] = tensor
    return res

def checkAllEqual(tp):
    """
    Check if all the elements in a tuple is the same.

    Parameters
    ----------
    tp : tuple of any

    Returns 
    bool
        Whether the values in tp are all the same.
    """
    if (len(tp) == 0):
        return True
    for x in list(tp):
        if (x != tp[0]):
            return False
    return True

def errorMessage(err, location = None):
    """
    Generate a standard error message.

    Parameters
    ----------
    err : str
        The error message.
    location : str, optional
        Where the error happens. E.g. CTL.funcs.funcs.errorMessage
    
    Returns
    -------
    str
        The generated error message.
    """
    if (location is None):
        return "Error: {}".format(err)
    else:
        return "Error in {}: {}".format(location, err)

def warningMessage(warn, location = None):
    """
    Generate a standard warning message.

    Parameters
    ----------
    warn : str
        The warning message.
    location : str, optional
        Where the warning happens. E.g. CTL.funcs.funcs.warningMessage
    
    Returns
    -------
    str
        The generated warning message.
    """
    if (location is None):
        return "Warning: {}".format(warn)
    else:
        return "Warning in {}: {}".format(location, warn)

def diagonalNDTensor(a, dim):
    """
    Create a diagonal N-dimensional tensor.

    Parameters
    ----------
    a : array of floats
        The values that will be on the diagonal of the tensor.
    dim : int
        The dimension of the output tensor.

    Returns
    -------
    ndarray
        A diagonal tensor of dim dimension, and contains the values of a in order on its diagonal.

    """
    l = a.shape[0]
    res = xplib.xp.zeros(tuple([l] * dim))
    res[xplib.xp.diag_indices(l, dim)] = a
    # for i in range(l):
    # 	res[tuple([i] * dim)] = a[i]
    return res

def indexTupleToStr(idx):
    """
    Generate a string contains all the lowercase letters corresponding to given index list.

    Parameters
    ----------
    idx : list of int
        A list of indices, each should be in [0, 26).
    
    Returns
    -------
    str
        A string that corresponding to the indices in idx. E.g. if input [1, 3, 5], then return "ace"
    """
    labelList = 'abcdefghijklmnopqrstuvwxyz'
    if (not isinstance(idx, tuple)):
        raise ValueError(errorMessage('indexTupleToStr requires a tuple as idx, {} gotten.'.format(idx)))
    
    for x in idx:
        if (x >= 26):
            raise ValueError(errorMessage('indexTupleToStr cannot transfer index >= 26 for index {}.'.format(idx)))
    
    return ''.join([labelList[x] for x in idx])

def ndEye(n, l):
    """
    Create a n-dimensional eye tensor.

    Parameters
    ----------
    n : int
        The dimension of the returned tensor.
    l : int
        The size of each dimension of the returned tensor.
    
    Returns
    -------
    ndarray
        A tensor of shape (l, l, ... l), all values on the main diagonal is 1.
    """
    res = xplib.xp.zeros(tuple([l] * n))
    if (n == 1):
        res[0] = 1.0
    else:
        res[xplib.xp.diag_indices(l, n)] = 1
    return res

def nonZeroElementN(s, eps = 1e-10):
    """
    Check the number of elements in an array.

    Parameters
    ----------
    s : array of float
    
    eps : float, default 1e-10
        The threshold over which we will decide a value as non-zero.

    Returns
    -------
    int
        The number of elements whose absolute value is over eps.
    """
    res = xplib.xp.count_nonzero(xplib.xp.abs(s) > eps)

    # polyfill for cupy, since cupy will consider this result as 0-d scalar
    if (type(res) != int):
        res = int(xplib.xp.asnumpy(res)[()])
    return res

def rightDiagonalProduct(a, diag):
    """
    Calculate the product of a matrix and a diagonal matrix, with broadcast.

    Parameters
    ----------
    a : 2D ndarray of float
    
    diag : array of float
        The diagonal elements of the diagonal matrix.

    Returns
    2D ndarray of float
        Matrix (a @ diag).
    """
    return a * diag 
def leftDiagonalProduct(a, diag):
    """
    Calculate the product of a diagonal matrix and a matrix, with broadcast.

    Parameters
    ----------
    a : 2D ndarray of float
    
    diag : array of float
        The diagonal elements of the diagonal matrix.

    Returns
    -------
    2D ndarray of float
        Matrix (diag @ a).
    """
    return (a.T * diag).T

def combineName(namesList, givenName = None):
    """
    Generate a name from a set of names, and an optional pre-given name.

    Parameters
    ----------
    namesList : list of str
        The list of names to be combined.
    givenName : str, optional
        A pre-given name. If not given, then we combine the names in namesList.

    Returns
    -------
    str
        The combined name if givenName is None, or just the pre-given name.
    """
    if (givenName is None):
        return '|'.join(namesList)
    else:
        return givenName

def isRealNumber(a):
    """
    Decide whether a is a number.

    Parameters
    ----------
    a : any

    Returns
    -------
    bool
        Whether a is a number(int, float) or not.
    """
    return (type(a) == int) or (type(a) == float)

def pairIterator(a):
    """
    Generate pairs (a[i], a[j]) of values in a, i < j.

    Parameters
    ----------
    a : list of any

    Yields
    -------
    tuple of any
        Tuples (a[i], a[j]) where (i, j) in order.
    """
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            yield a[i], a[j]

    return

def priorDataType(dtype1, dtype2):
    """
    Decide the data type of the output of two input types.

    Parameters
    ----------
    dtype1, dtype2 : xplib.xp.dtype

    Returns
    -------
    xplib.xp.dtype
        The output type with the highest accuracy of dtype1 and dtype2, and is complex if any of them is complex.
    """

    location = 'CTL.funcs.funcs.priorDataType'
    isFloat1 = xplib.xp.issubdtype(dtype1, xplib.xp.floating)
    isFloat2 = xplib.xp.issubdtype(dtype2, xplib.xp.floating)

    length1 = dtype1.itemsize
    length2 = dtype2.itemsize

    acc1, acc2 = length1, length2 
    if not isFloat1:
        acc1 = acc1 // 2
    if not isFloat2:
        acc2 = acc2 // 2
    
    isFloat = isFloat1 and isFloat2
    acc = max(acc1, acc2)

    if isFloat:
        if acc == 2:
            return xplib.xp.float16
        elif acc == 4:
            return xplib.xp.float32
        elif acc == 8:
            return xplib.xp.float64
        elif acc == 16:
            return xplib.xp.float128
        else:
            raise ValueError(errorMessage(err = 'accuracy {}(byte) is not compatible with float types of numpy(need [2, 4, 8, 16]).'.format(acc), location = location))
    
    else:
        if acc == 4:
            return xplib.xp.complex64
        elif acc == 8:
            return xplib.xp.complex128
        elif acc == 16:
            return xplib.xp.complex256
        else:
            raise ValueError(errorMessage(err = 'accuracy {}(byte) is not compatible with complex types of numpy(need [4, 8, 16]).'.format(acc), location = location))

def generateIndices(elemList, elems):
    """
    Generate the indices of label according to labelList. If some element appears more than once, then take indices in order

    Parameters
    ----------
    elemList : list of any
        Index of which is asked.
    elems : list of any
        For each element in elems, return its index in elemList.
    
    Returns
    -------
    list of int or None
        Indices for all elements in elems. None for elements not found. 
        We do not take -1 for elements not found, since this may be misread as the last element of list.
    """

    currentIndex = dict()
    res = []
    for elem in elems:
        startIndex = currentIndex.get(elem, -1) + 1
        if elem not in elemList[startIndex:]:
            res.append(None) 
        else:
            index = elemList[startIndex:].index(elem) + startIndex
            res.append(index)
            currentIndex[elem] = index
    
    return res

