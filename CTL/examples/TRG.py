from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.contract import contractTensors
from CTL.tensor.tensorFactory import makeTriangleTensor, makeSquareTensor
from CTL.tensor.contract.contractExp import triangleContractFTN, makeTriangleTensorDict, triangleTensorTrace
from CTL.tensor.contract.contractExp import squareContractOutFTN
from CTL.funcs.decompose import SVDDecomposition
from CTL.tensornetwork.tensordict import TensorDict
import numpy as np

# Tensor Renormalization Group

# triangle lattice TRG, PRL 99, 120601 (2007)

class TriangleTRG:

    def __init__(self, alpha, chi = 16):
        self.alpha = alpha 
        self.iterateFTN = None
        self.chi = chi

        self.prepareInitialTensor()
    
    def prepareInitialTensor(self):
        a = np.zeros((2, 2, 2), dtype = np.float64)
        a[(0, 0, 0)] = 1.0
        a[(0, 0, 1)] = a[(0, 1, 0)] = a[(1, 0, 0)] = self.alpha 
        self.a = makeTriangleTensor(a)
        self.b = makeTriangleTensor(a)

        self.a.degreeOfFreedom = 1
        self.b.degreeOfFreedom = 1

        self.aNorms = []
        self.bNorms = []
        self.errors = []

        self.aArchive = []
        self.bArchive = []

        self.appendToArchive()

    def appendToArchive(self):
        self.normalizeTensors()
        self.aArchive.append(self.a.copy())
        self.bArchive.append(self.b.copy())

    def normalizeTensors(self):
        aNorm = self.a.norm()
        self.a.a /= aNorm 
        bNorm = self.b.norm()
        self.b.a /= bNorm 
        self.aNorms.append(aNorm)
        self.bNorms.append(bNorm)

    def iterate(self):
        # print('iterate:')
        if (self.iterateFTN is None):
            self.iterateFTN = triangleContractFTN()

        self.a.addTensorTag('a')
        self.b.addTensorTag('b')

        dof = self.a.degreeOfFreedom
        makeLink(self.a.getLeg('a-1'), self.b.getLeg('b-1'))
        iTensor = contractTensors(self.a, self.b)

        a2Dim, b2Dim, a3Dim, b3Dim = iTensor.shapeOfLabels(['a-2', 'b-2', 'a-3', 'b-3'])

        iMat = iTensor.toMatrix(rows = ['a-2', 'b-2'], cols = ['a-3', 'b-3'])
        u, v, error = SVDDecomposition(iMat, self.chi)
        self.errors.append(error)
        # print(u.shape, v.shape)
        # print(iTensor.shape)

        u = np.reshape(u, (a2Dim, b2Dim, u.shape[1]))
        v = np.reshape(v, (a3Dim, b3Dim, v.shape[1]))

        uTensor = makeTriangleTensor(u, labels = ['2', '3', '1'])
        vTensor = makeTriangleTensor(v, labels = ['2', '3', '1'])

        self.a = self.iterateFTN.contract(makeTriangleTensorDict(uTensor))
        self.b = self.iterateFTN.contract(makeTriangleTensorDict(vTensor))

        self.a.degreeOfFreedom = dof * 3
        self.b.degreeOfFreedom = dof * 3
        
        # self.normalizeTensors()
        self.appendToArchive()

    def logZDensity(self):
        # print(self.aArchive, self.bArchive)
        # logZ / degreeOfFreedom
        accumulateLogZ = 0.0
        res = []
        stepN = len(self.aArchive)
        # print(stepN, len(self.aNorms))
        for i in range(stepN):
            dof = self.aArchive[i].degreeOfFreedom * self.bArchive[i].degreeOfFreedom
            accumulateLogZ += np.log(self.aNorms[i] * self.bNorms[i]) / dof
            TNTrace = triangleTensorTrace(self.aArchive[i], self.bArchive[i])
            currLogZ = accumulateLogZ + np.log(TNTrace.single()) / dof
            # contraction: 1A + 1B
            res.append(currLogZ)

        return res

class SquareTRG:

    def __init__(self, a, b = None, chi = 16):
        assert (a is not None), "Error: SquareTRG must be initialized with at least one tensor."
        self.iterateFTN = None
        self.chi = chi
        if (isinstance(a, Tensor)):
            self.a = a.copy()
        else:
            self.a = makeSquareTensor(a)

        if (self.a.degreeOfFreedom is None):
            self.a.degreeOfFreedom = 1
        if (b is not None):
            if (isinstance(b, Tensor)):
                self.b = b.copy()
            else:
                self.b = makeSquareTensor(b)
            
            self.iterate()
        else:
            self.b = self.a

        self.aNorms = []
        self.errors = []

        self.aArchive = []

        self.appendToArchive()

    def appendToArchive(self):
        self.normalizeTensors()
        self.aArchive.append(self.a.copy())

    def normalizeTensors(self):
        aNorm = self.a.norm()
        self.a.a /= aNorm 
        self.aNorms.append(aNorm)

    # def initialIterate(self):
    #     # contract a, b so that we only hold one tensor
    #     pass

    def iterate(self):
        # make SVD decomposition on square tensors
        # do for both a and b are different and the same

        if (self.iterateFTN is None):
            self.iterateFTN = squareContractOutFTN()

        dof = self.a.degreeOfFreedom
        # makeLink(self.a.getLeg('a-1'), self.b.getLeg('b-1'))
        # iTensor = contractTensors(self.a, self.b)

        # a2Dim, b2Dim, a3Dim, b3Dim = iTensor.shapeOfLabels(['a-2', 'b-2', 'a-3', 'b-3'])

        # aMat = iTensor.toMatrix(rows = ['a-2', 'b-2'], cols = ['a-3', 'b-3'])

        # a: at ul and dr
        vDim, hDim = self.a.shapeOfLabels(['u', 'l'])
        aMat = self.a.toMatrix(rows = ['u', 'l'], cols = ['d', 'r'])
        bMat = self.b.toMatrix(rows = ['u', 'r'], cols = ['d', 'l'])

        uA, vA, errorA = SVDDecomposition(aMat, self.chi)
        uB, vB, errorB = SVDDecomposition(bMat, self.chi)

        # uA: u, l, o as dr
        # vA: d, r, o as ul
        dr = Tensor(shape = (vDim, hDim, uA.shape[1]), labels = ['u', 'l', 'o'], data = uA)
        ul = Tensor(shape = (vDim, hDim, vA.shape[1]), labels = ['d', 'r', 'o'], data = vA)
        dl = Tensor(shape = (vDim, hDim, uB.shape[1]), labels = ['u', 'r', 'o'], data = uB)
        ur = Tensor(shape = (vDim, hDim, vB.shape[1]), labels = ['d', 'l', 'o'], data = vB)

        self.a = self.iterateFTN.contract(TensorDict({'ul': ul, 'ur': ur, 'dl': dl, 'dr': dr}))
        self.a.degreeOfFreedom = dof * 2
        self.b = self.a 
        self.appendToArchive()
        self.errors.append((errorA, errorB))

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
            # print(self.aNorms[i], TNTrace, dof)
            currLogZ = accumulateLogZ + np.log(TNTrace) / dof
            # contraction: 1A + 1B
            res.append(currLogZ)
            # print(currLogZ)

        return np.array(res)



# Given two tensors A, B
# svd on both, then contract on a square to obtain new A, B