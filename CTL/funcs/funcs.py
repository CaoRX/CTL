from collections import Counter
import numpy as np 
import string
import random
import math
from copy import deepcopy

def deprecatedFuncWarning(funcName, fileName = None, newFuncName = None):
	if (fileName is None):
		fileNameInfo = ''
	else:
		fileNameInfo = 'in {} '.format(fileName)
	if (newFuncName is None):
		print('Warning: {} {}has been deprecated. This function should not be used anywhere.'.format(funcName, fileNameInfo))
	else:
		print('Warning: {} {}has been deprecated. Please use {} instead.'.format(funcName, fileNameInfo, newFuncName))

def listDifference(total, used):
	ret = list(total)
	for x in used:
		ret.remove(x)
	return ret

def deleteAllInUsed(total, used):
	usedSet = set(used)
	ret = list(total)
	return [x for x in ret if not (x in usedSet)]

def tupleRemoveByIndex(initTuple, indexList):
	initList = list(initTuple)
	indexSet = set(indexList)
	resList = []
	for i in range(len(initList)):
		if not (i in indexSet):
			resList.append(initList[i])
	return tuple(resList)

def tupleProduct(shapeTuple):
	res = 1
	for x in shapeTuple:
		res *= x
	return res

def spinChainProductSum(spins):
	res = 0.0
	n = len(spins)
	for i in range(n):
		res += spins[i] * spins[i + 1 - n]
	return res

def compareLists(a, b):
	return Counter(a) == Counter(b)

def commonElements(x, y):
	xc = Counter(x)
	yc = Counter(y)
	return list((xc & yc).elements())

def listSymmetricDifference(x, y):
	return list(set(x) ^ set(y))

def intToBitTuple(x, dim):
	res = []
	for _ in range(dim):
		res.append(x % 2)
		x = x // 2

	return tuple(res)

def safeMod(x, m):
	return ((x % m) + m) % m

def intToDBaseTuple(x, dim, D):
	res = []
	for _ in range(dim):
		res.append(x % D)
		x = x // D
	return tuple(res)

def triangle_weight(sm, J):
	# print('Warning: triangle_weight in funcs/funcs.py has been deprecated. This function should not be used anywhere.')
	deprecatedFuncWarning(funcName = 'triangle_weight', fileName = 'funcs/funcs')
	if (sm == 0):
		return 1
	if (sm == 2):
		return np.exp(-J)
	return 0.0

def loadData(filename):
	f = open(filename, 'r')
	data = np.array([np.array([np.float(x) for x in line.strip().split(' ')]) for line in f])
	return data

def randomString(n = 10):
	return ''.join(random.choice(string.ascii_letters) for i in range(n))

def make_iden(shape):
	# print('Warning: make_iden in funcs/funcs.py has been deprecated. This function should not be used anywhere.')
	deprecatedFuncWarning(funcName = 'make_iden', fileName = 'funcs/funcs')
	assert (len(shape) == 2), 'make_iden can only make 2D matrix, but got shape {}'.format(shape)
	res = np.zeros(shape, dtype = np.float64)
	for i in range(min(shape)):
		res[i][i] = 1.0

	return res

def identityError(a):
	return np.linalg.norm(a - np.eye(a.shape[0], a.shape[1]))

def diagError(a):
	return np.linalg.norm(a - np.diag(np.diagonal(a))) / np.linalg.norm(a)

def checkIdentity(a, eps = 1e-10):
	return identityError(a) < eps

def symmetricError(a):
	return np.linalg.norm(a - np.transpose(a)) / np.linalg.norm(a)
def transposeConjugate(a):
	# transpose conjugate of matrix a
	return np.conjugate(np.transpose(a))
def aDaggerAProduct(a):
	return np.matmul(transposeConjugate(a), a)
def aADaggerProduct(a):
	return np.matmul(a, transposeConjugate(a))

def projectorError(a):
	print('project error of matrix with shape {}'.format(a.shape))
	return identityError(aDaggerAProduct(a))
def get_diff_error(a, b):
	deprecatedFuncWarning(funcName = 'get_diff_error', fileName = 'funcs/funcs')
	return np.linalg.norm(a ** 2) + np.linalg.norm(b ** 2) - np.linalg.norm(a * b) * 2

def matDiffError(a, b):
	return np.linalg.norm(a - b) / np.linalg.norm(a)

def randomArray(shape):
	return 2.0 * np.random.random(shape) - 1.0

def print_array(a, eps = 1e-15):
	deprecatedFuncWarning(funcName = 'print_array', fileName = 'funcs/funcs')
	a_copy = a.copy()
	a_non_zero = a_copy[np.abs(a_copy) > eps]
	return a_non_zero

def assertInSet(x, xSet, name):
	assert (x in xSet), 'Error: {} should be one of {}, but {} gotten.'.format(name, xSet, x)

def dealParams(paramsDict, paramsNeed, paramsPassed):
	for key in paramsPassed:
		if (key in paramsDict):
			paramsDict[key] = paramsPassed[key]

	for key in paramsNeed:
		if (paramsDict[key] is None):
			return False, key

	return True, None

def paramsFuncMaker(paramsDict, paramsNeed, initialFuncs, initialFuncsName):

	paramsDictLocal = deepcopy(paramsDict)
	paramsNeedLocal = deepcopy(paramsNeed)

	def paramsFunc(paramsPassed):
		# print('paramsDict = {}'.format(paramsDictLocal))
		paramsDictCopy = deepcopy(paramsDictLocal)
		paramsNeedCopy = deepcopy(paramsNeedLocal)
		paramsOk, errorMessage = dealParams(paramsDict = paramsDictCopy, paramsNeed = paramsNeedCopy, paramsPassed = paramsPassed)

		assert (paramsOk), 'Error: parameter {} is needed in function {} but not passed.\ndefault parameters = {}\nparameters needed = {}.'.format(errorMessage, initialFuncsName, paramsDictLocal, paramsNeedLocal)
		return initialFuncs(**paramsDictCopy)

	return paramsFunc

def paramsFuncPackager(paramsDict, paramsNeed, initialFuncs, initialFuncsName, paramsName = 'params'):

	# work for methods with parameters, and a parameter called params
	# initialFuncs needs all params in the paramsDict as params
	# however, we want to set a "default" params, that should be done by this packager

	# return a function which takes just the same parameters as initialFuncs, including a "params"
	# however, when taking small set of params, we should still generate the whole paramsDict with default values here

	paramsDictLocal = deepcopy(paramsDict)
	paramsNeedLocal = deepcopy(paramsNeed)

	def paramsFunc(**kwargs):
		# print('paramsDict = {}'.format(paramsDictLocal))
		paramsDictCopy = deepcopy(paramsDictLocal)
		paramsNeedCopy = deepcopy(paramsNeedLocal)
		if (paramsName in kwargs) and (not (kwargs[paramsName] is None)):
			paramsOk, errorMessage = dealParams(paramsDict = paramsDictCopy, paramsNeed = paramsNeedCopy, paramsPassed = kwargs[paramsName])
			assert (paramsOk), 'Error: parameter {} is needed in function {} but not passed.\ndefault parameters = {}\nparameters needed = {}.'.format(errorMessage, initialFuncsName, paramsDictLocal, paramsNeedLocal)

		kwargs[paramsName] = paramsDictCopy
		return initialFuncs(**kwargs)

	return paramsFunc

def getIndexDict(idx, labels, shape):
	res = dict([])
	for i in range(len(shape) - 1, -1, -1):
		res[labels[i]] = idx % shape[i]
		idx //= shape[i]
	return res

def getNewString(knownSet):
	res = randomString()
	while (res in knownSet):
		res = randomString()
	return res

def divideIntoKParts(n, k):
	if (k == 1):
		yield [n]
		return 

	for i in range(n + 1):
		for partAfter in divideIntoKParts(n - i, k - 1):
			yield [i] + partAfter
	return

def calculateDivisionTimes(division):
	res = 1
	n = sum(division)
	for p in division:
		res *= math.comb(n, p)
		n -= p
	return res

def getValidValues(values, threshold = 1e10):
	resX = []
	resY = []
	for i, value in enumerate(values):
		if (np.abs(value) < threshold):
			resX.append(i)
			resY.append(value)
	return resX, resY

def normalizeInfinity(a):
	return a / np.max(np.abs(a))

def binarySearch(left, right, largeEnough, eps):
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
	res = []
	for name in names:
		assert (name in listDict), 'Error: name asked {} does not exist in listDict with keys {}.'.format(name, list(listDict.keys()))
		res = res + listDict[name]
	return set(res)

def getCutoff(s, threshold):
	for i in range(len(s)):
		if (s[i] < threshold):
			return i
	return len(s)

def partialInverse(s, cutoff):
	res = np.zeros(len(s))
	for i in range(cutoff):
		res[i] = 1.0 / s[i]
	return res

def oppositeDirection(x):
	res = ""
	for i in range(len(x)):
		res = res + oppositeSingleDirection(x[i])
	return res

def oppositeSingleDirection(x):
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
	return lambda x: (x > low and x < high)

def mapToDict(dictShape, x):
	res = dict([])
	for key in dictShape:
		res[key] = x
	return res

class LinearRegressionModel:

	def __init__(self, x, y):
		self.fit(x, y)

	def fit(self, x, y):
		self.x = np.array(deepcopy(x))
		self.y = np.array(deepcopy(y))

		averageX = np.average(self.x)
		averageY = np.average(self.y)
		averageXX = np.average(self.x ** 2)
		averageYY = np.average(self.y ** 2)
		averageXY = np.average(self.x * self.y)

		self.k = (averageXY - averageX * averageY) / (averageXX - averageX ** 2)
		self.b = averageY - self.k * averageX 
		self.r = (averageXY - averageX * averageY) / np.sqrt((averageXX - averageX ** 2) * (averageYY - averageY ** 2))

	def predict(self, x):
		return self.k * x + self.b

def tupleSwap(tp):
	a, b = tp
	return (b, a)

def floatEqual(a, b, eps = 1e-7):
	return np.abs(a - b) < eps

