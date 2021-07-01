import numpy as np 
import CTL.funcs.funcs as funcs
from CTL.tensor.tensor import Tensor
from CTL.tensor.diagonalTensor import DiagonalTensor
import warnings

def shareBonds(ta, tb):
	bonds = []
	for leg in ta.legs:
		anotherLeg = leg.anotherSide()
		if (anotherLeg is not None) and (anotherLeg in tb.legs):
			bonds.append(leg.bond)
			if (leg == leg.bond.legs[1]):
				leg.bond.legs = funcs.tupleSwap(leg.bond.legs)
	return bonds

# how to contract diagonal tensors?
# 0. if both diagonal: then just do product over same index, and combine legs O(n)
# 1. if one is diagonal: then we take all the bonds linked to it
# for each output element: we can decide it in O(1) time
# so the complexity is just the shape of output tensor

def contractTwoTensors(ta, tb, bonds = None, outProductWarning = True):
	# contract between bonds(if not, then find the shared legs)
	# this requires that: tensor contraction happens in-place
	# after we prepare some tensor, we must create tensors to make links

	if (not ta.diagonalFlag) and (tb.diagonalFlag):
		return contractTwoTensors(tb, ta, bonds = bonds, outProductWarning = outProductWarning)
	if (bonds is None):
		bonds = shareBonds(ta, tb)
	
	if (len(bonds) == 0):
		if (outProductWarning):
			warnings.warn('{} and {} do not share same label, do out product'.format(ta, tb), RuntimeWarning)

		# aMatrix = ta.toMatrix(rows = ta.legs, cols = [])
		# bMatrix = tb.toMatrix(rows = [], cols = tb.legs)
		# data = np.matmul(aMatrix, bMatrix)

		labels = ta.labels + tb.labels 
		shape = ta.shape + tb.shape
		legs = ta.legs + tb.legs

		if (ta.diagonalFlag and tb.diagonalFlag):
			return DiagonalTensor(labels = labels, data = ta.a * tb.a, shape = shape, legs = legs)
		else:
			return Tensor(labels = labels, data = np.multiply.outer(ta.a, tb.a), shape = shape, legs = legs)

		# aVector = ta.toVector()
		# bVector = tb.toVector()
		# data = np.outer(aVector, bVector)
		# data = np.reshape(data, shape)

		# return Tensor(labels = labels, data = data, legs = legs)

	contractALegs = [bond.legs[0] for bond in bonds]
	contractBLegs = [bond.legs[1] for bond in bonds]

	taRemainLegs = ta.complementLegs(contractALegs)
	tbRemainLegs = tb.complementLegs(contractBLegs)
	newLegs = taRemainLegs + tbRemainLegs 
	newShape = tuple([leg.dim for leg in newLegs])

	if (ta.diagonalFlag) and (tb.diagonalFlag):
		# return a diagonal tensor
		return DiagonalTensor(shape = newShape, data = ta.a * tb.a, legs = newLegs)
	
	if (ta.diagonalFlag):
		# then tb is not diagonal tensor
		# 1. calculate the core with broadcast
		# 2. calculate the real tensor with np.outer
		# how to broadcast?

		# we need to broadcast from the end(instead of the first dimension)
		# so we need to make the contract legs to the end
		# then we need to take diagonal from these dimensions

		tb.moveLegsToFront(tbRemainLegs)
		dim = len(contractBLegs)
		l = contractBLegs[0].dim
		# print('contract A legs: {}'.format(len(contractALegs)))
		# print('contract B legs: {}'.format(len(contractBLegs)))
		remADim = len(taRemainLegs)

		einsumStr = '...' + ('j' * dim) + '->...j'
		# print('ta.a = {}, tb.a = {}'.format(ta.a, tb.a))
		data = np.einsum(einsumStr, tb.a) * ta.a
		# print('einsum str = {}'.format(einsumStr))
		# print('einsum b = {}'.format(np.einsum(einsumStr, tb.a)))
		# print('data = {}'.format(data))

		if (remADim == 0):
			newData = np.sum(data, axis = -1)
		else:
			# print('remADim = {}, data.shape = {}'.format(remADim, data.shape))
			newData = np.zeros(data.shape + (data.shape[-1],) * (remADim - 1), dtype = data.dtype)
			# print('newData.shape = {}'.format(newData.shape))
			
			einsumDiagStr = '...' + ('j' * (remADim)) + '->...j'
			np.einsum(einsumDiagStr, newData)[...] = data
			# newData = np.einsum(eimsumDiagStr, data)
			# print(funcs.ndEye(remADim - 1, l))
			# newData = np.multiply.outer(data, funcs.ndEye(remADim - 1, l))
			# print('newData = {}'.format(newData))
		# print('newData = {}'.format(newData))
		# print('shape = {}, data = {}, legs = {}'.format(newShape, newData, newLegs))

		newLegs = tbRemainLegs + taRemainLegs 
		newShape = tuple([leg.dim for leg in newLegs])
		return Tensor(shape = newShape, data = newData, legs = newLegs)

	dataA = ta.toMatrix(rows = None, cols = contractALegs)
	dataB = tb.toMatrix(rows = contractBLegs, cols = None)
	newData = np.matmul(dataA, dataB)

	newData = np.reshape(newData, newShape)

	return Tensor(shape = newShape, data = newData, legs = newLegs)

# def contractTwoTensorsByLabel(ta, tb, labels): # not in-place
# 	labelA, labelB = labels
# 	if (isinstance(labelA, str)):
# 		label_a = [labelA]
# 	if (isinstance(labelB, str)):
# 		label_b = [labelB]

# 	assert (len(labelA) == len(labelB)), 'must contract same number of labels, {} & {} got'.format(len(labelA), len(labelB))

# 	aCopy = ta.copy()
# 	bCopy = tb.copy()

# 	bond_names = makeLink(aCopy, bCopy, (label_a, label_b))
# 	res = contractTwoTensors(ta_cp, tb_cp, cr = bond_names)
# 	return res.copy()

# def self_trace_square(tensor_tuple):
# 	deprecateFuncWarning(funcName = 'self_trace_square', fileName = 'tensor/tensor_contract.py')
# 	t = tensor_tuple[0]
# 	return t.selfContract(rows = ['l', 'u'], cols = ['r', 'd'])

# def square_contract_general(lu, ld, ru, rd):
# 	makeSquareLink(lu = lu, ld = ld, ru = ru, rd = rd)
# 	lu.renameLabel('u', 'u1')
# 	lu.renameLabel('l', 'l2')
# 	ru.renameLabel('u', 'u2')
# 	ru.renameLabel('r', 'r1')
# 	rd.renameLabel('r', 'r2')
# 	rd.renameLabel('d', 'd1')
# 	ld.renameLabel('d', 'd2')
# 	ld.renameLabel('l', 'l1')

# 	l = contractTwoTensors(lu, ld)
# 	r = contractTwoTensors(ru, rd)
# 	t = contractTwoTensors(l, r)
# 	t.outerProduct(['u1', 'u2'], 'u')
# 	t.outerProduct(['d2', 'd1'], 'd')
# 	t.outerProduct(['l1', 'l2'], 'l')
# 	t.outerProduct(['r2', 'r1'], 'r')
# 	if not (lu.degreeOfFreedom is None):
# 		t.degreeOfFreedom = lu.degreeOfFreedom * 4
# 	return t.copy()

# def square_contract(a, symmetry_eps = 1e-10):
# 	sameFlag = True
# 	if (isinstance(a, tuple)):
# 		assert (len(a) == 2), 'square contract can take at most 2 tensors, but {} gotten.'.format(len(a))
# 		ta, tb = a 
# 		alu, ard = ta.copyN(2)
# 		aru, ald = tb.copyN(2)
# 		sameFlag = False
# 	else:
# 		alu, ald, aru, ard = a.copyN(4)
# 	if (sameFlag and (symmetryError(a) < symmetry_eps)):
# 		return square_contract_symmetric(a)
# 	else:
# 		return square_contract_general(lu = alu, ld = ald, ru = aru, rd = ard)
	
# def square_contract_symmetric(a): 
# 	print('symmetric contracting...')
# 	# used when we want a symmetric tensor when exchanging horizontal and vertical indices
# 	# generally, using this function means we have a special boundary condition
# 	if (isinstance(a, tuple)):
# 		assert (len(a) == 2), 'square contract can take at most 2 tensors, but {} gotten.'.format(len(a))
# 		ta, tb = a 
# 		alu, ard = ta.copyN(2)
# 		aru, ald = tb.copyN(2)
# 	else:
# 		alu, ald, aru, ard = a.copyN(4)
# 	makeSquareLink(lu = alu, ld = ald, ru = aru, rd = ard)
# 	alu.renameLabel('u', 'u1')
# 	alu.renameLabel('l', 'l2')
# 	aru.renameLabel('u', 'u2')
# 	aru.renameLabel('r', 'r1')
# 	ard.renameLabel('r', 'r2')
# 	ard.renameLabel('d', 'd1')
# 	ald.renameLabel('d', 'd2')
# 	ald.renameLabel('l', 'l1')

# 	l = contractTwoTensors(alu, ald)
# 	r = contractTwoTensors(aru, ard)
# 	t = contractTwoTensors(l, r)
# 	t.outerProduct(['u1', 'u2'], 'u')
# 	t.outerProduct(['d1', 'd2'], 'd')
# 	t.outerProduct(['l1', 'l2'], 'l')
# 	t.outerProduct(['r1', 'r2'], 'r')
# 	if not (a.degreeOfFreedom is None):
# 		t.degreeOfFreedom = a.degreeOfFreedom * 4
# 	return t.copy()

# def square_contract_trace(a):
# 	if (isinstance(a, Tensor)):
# 		alu, ald, aru, ard = a.copyN(4)
# 	else:
# 		alu = a['lu'].copy()
# 		aru = a['ru'].copy()
# 		ald = a['ld'].copy()
# 		ard = a['rd'].copy()
	
# 	makeSquareLink(lu = alu, ld = ald, ru = aru, rd = ard)
# 	makeLink(alu, ald, ('u', 'd'))
# 	makeLink(aru, ard, ('u', 'd'))
# 	makeLink(alu, aru, ('l', 'r'))
# 	makeLink(ald, ard, ('l', 'r'))

# 	l = contractTwoTensors(alu, ald)
# 	r = contractTwoTensors(aru, ard)
# 	t = contractTwoTensors(l, r)
# 	return t.a
