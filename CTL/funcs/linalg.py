import numpy as np 

def solveEnv(aMat, chi, threshold = 1e-10):
    eigenValuesEnv, eigenVectorsEnv = np.linalg.eigh(aMat)
    eigenExist = sum(eigenValuesEnv > threshold)
    # print(eigenValuesEnv)
    if (chi > eigenExist):
        error = 0.0
    else:
        error = 1.0 - np.sum(eigenValuesEnv[-chi:]) / np.sum(eigenValuesEnv)
    
    chi = min(chi, eigenExist)
    # if (chi > eigenVectorsEnv.shape[1]):
    #     return eigenVectorsEnv, 0.0
    # else:
    return np.flip(eigenVectorsEnv[:, -chi:], axis = 1), error
    # return eigenVectorsEnv[:, -chi:]