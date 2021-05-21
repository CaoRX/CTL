import numpy as np 
import CTL.funcs.funcs as funcs

def SVDDecomposition(a, chi):

    u, s, vh = np.linalg.svd(a)
    
    sLen = len(s)
    if (chi >= sLen):
        error = 0.0
        chi = sLen
    else:
        error = np.sqrt(np.sum(s[chi:] ** 2) / np.sum(s ** 2))

    # print('error = {}'.format(error))

    u = u[:, :chi]
    s = s[:chi]
    vh = vh[:chi]

    sqrtSMat = np.diag(np.sqrt(s))

    uRes = u @ sqrtSMat
    vRes = funcs.transposeConjugate(sqrtSMat @ vh)
    return uRes, vRes, error



