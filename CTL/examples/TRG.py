from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.link import makeLink
from CTL.tensor.contract.contract import contractTensors
from CTL.tensor.tensorFactory import makeTriangleTensor
from CTL.tensor.contract.contractExp import triangleContractFTN, makeTriangleTensorDict
from CTL.funcs.decompose import SVDDecomposition
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
        
        self.normalizeTensors()
        self.appendToArchive()

    def logZDensity(self):
        # logZ / degreeOfFreedom
        pass 



    #     ta, tb = self.tensor.copyN(2)
    #     makeLink('u', 'u', ta, tb)
    #     a = contractTensors(ta, tb)
        # self.tensor = triangleContract(a)

# Given two tensors A, B
# svd on both, then contract on a square to obtain new A, B