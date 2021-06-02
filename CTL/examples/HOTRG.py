import numpy as np 
from CTL.tensor.tensor import Tensor
from CTL.tensor.tensorFactory import makeSquareTensor
from CTL.tensor.contract.contractExp import squareHorizontalContractFTN, squareVerticalContractFTN
from CTL.tensor.contract.contractExp import HOTRGHorizontalContractFTN, HOTRGVerticalContractFTN
from CTL.tensor.contract.contractExp import makeSquareTensorDict
import CTL.funcs.funcs as funcs
import CTL.funcs.linalg as linalgFuncs

class HOTRG:

    # square lattice HOTRG
    # start with a single tensor, contract on square lattice
    # each time contract on one direction by two tensors, do HOSVD, then 

    def __init__(self, a, chiH = 16, chiV = None):
        assert (a is not None), "Error: HOTRG must be initialized with at least one tensor."
        self.horizontalIterateFTN = HOTRGHorizontalContractFTN()
        self.verticalIterateFTN = HOTRGVerticalContractFTN()
        self.horizontalProjectFTN = squareHorizontalContractFTN()
        self.verticalProjectFTN = squareVerticalContractFTN()
        self.chiH = chiH
        if (chiV is None):
            self.chiV = chiH
        else:
            self.chiV = chiV
        if (isinstance(a, Tensor)):
            self.a = a.copy()
        else:
            self.a = makeSquareTensor(a)

        if (self.a.degreeOfFreedom is None):
            self.a.degreeOfFreedom = 1

        self.aNorms = []
        self.errors = []

        self.aArchive = []
        self.directChoices = []
        self.projectors = []

        self.appendToArchive()

    def appendToArchive(self):
        self.normalizeTensors()
        self.aArchive.append(self.a.copy())

    def normalizeTensors(self):
        aNorm = self.a.norm()
        self.a.a /= aNorm 
        self.aNorms.append(aNorm)

    def directedIterateTrial(self, d):
        funcs.assertInSet(d, ['u', 'd', 'l', 'r'], 'direction')
        # make projector, and return the truncation error
        # for l: use M, M', calculate MM', return U_L
        if (d == 'l') or (d == 'r'):
            # horizontal project
            squareTensor = self.horizontalProjectFTN.contract(makeSquareTensorDict(self.a))
            # l, r
            # labels = [d, funcs.oppositeDirection(d)]
            # print('a = {}'.format(self.a))
            # print('squareTensor = {}'.format(squareTensor))
            squareMat = squareTensor.toMatrix(rows = [d], cols = [funcs.oppositeDirection(d)])
            # print(squareTensor)
            prjMat, error = linalgFuncs.solveEnv(squareMat, self.chiH)
            # prjTensor = Tensor(data = prjMat, shape = (chiH ** 2, prjMat.shape[1]), labels = ['i', 'o'])
        else:
            squareTensor = self.verticalProjectFTN.contract(makeSquareTensorDict(self.a))
            # print('a = {}'.format(self.a))
            # print('squareTensor = {}'.format(squareTensor))
            squareMat = squareTensor.toMatrix(rows = [d], cols = [funcs.oppositeDirection(d)])
            prjMat, error = linalgFuncs.solveEnv(squareMat, self.chiV)
            # prjTensor = Tensor(data = prjMat, shape = (chiV ** 2, prjMat.shape[1]), labels = ['i', 'o'])

        # envTrace = np.trace(squareMat)
        # envTraceApprox = np.trace(prjMat.T @ squareMat @ prjMat)
        # error = (envTrace - envTraceApprox) / envTrace 
        # deltaTensor = (prjMat.T @ squareMat @ prjMat) - squareMat 
        # error = np.linalg.norm(deltaTensor) / np.linalg.norm(squareMat)
        return {'error': error, 'projectTensor': prjMat}

    def directedIterate(self, d, prjTensor):
        funcs.assertInSet(d, ['u', 'd', 'l', 'r'], 'direction')
        # use the real projector for iteration
        chiH = self.a.shapeOfLabel('l')
        chiV = self.a.shapeOfLabel('u')
        if (d == 'l') or (d == 'r'):
            # given U_L: 
            # the prjTensor
            lTensor = Tensor(data = prjTensor, shape = (chiH, chiH, prjTensor.shape[1]), labels = ['u', 'd', 'o'])
            rTensor = Tensor(data = funcs.transposeConjugate(prjTensor), shape = (prjTensor.shape[1], chiH, chiH), labels = ['o', 'u', 'd'])
            if (d == 'r'):
                lTensor, rTensor = rTensor, lTensor 
            self.a = self.horizontalIterateFTN.contract({'u': self.a, 'd': self.a, 'l': lTensor, 'r': rTensor})
        else:
            uTensor = Tensor(data = prjTensor, shape = (chiV, chiV, prjTensor.shape[1]), labels = ['l', 'r', 'o'])
            dTensor = Tensor(data = funcs.transposeConjugate(prjTensor), shape = (prjTensor.shape[1], chiV, chiV), labels = ['o', 'l', 'r'])
            if (d == 'd'):
                uTensor, dTensor = dTensor, uTensor
            self.a = self.verticalIterateFTN.contract({'u': uTensor, 'd': dTensor, 'l': self.a, 'r': self.a})
            
    def iterate(self):
        # choose among 4 directions

        dof = self.a.degreeOfFreedom

        directions = ['u', 'd', 'l', 'r']
        minimumError = -1.0
        projectTensor = None 
        bestDirection = None 
        
        for d in directions: 
            projector = self.directedIterateTrial(d)
            if (minimumError < 0) or (projector['error'] < minimumError):
                minimumError = projector['error']
                bestDirection = d
                projectTensor = projector['projectTensor']
        
        self.directedIterate(bestDirection, projectTensor)
        self.a.degreeOfFreedom = dof * 2
        self.appendToArchive()
        self.errors.append(minimumError)
        self.directChoices.append(bestDirection)
        self.projectors.append(projectTensor)

    def logZDensity(self):
        # only a will be considered
        # for i in range(len(self.aArchive)):
        accumulateLogZ = 0.0
        res = []
        stepN = len(self.aArchive)
        # print(stepN, len(self.aNorms))
        for i in range(stepN):
            dof = self.aArchive[i].degreeOfFreedom
            accumulateLogZ += np.log(self.aNorms[i]) / dof
            # TNTrace = triangleTensorTrace(self.aArchive[i], self.bArchive[i])
            TNTrace = self.aArchive[i].trace(rows = ['u', 'l'], cols = ['d', 'r'])
            currLogZ = accumulateLogZ + np.log(TNTrace) / dof
            # contraction: 1A + 1B
            res.append(currLogZ)

        return np.array(res)