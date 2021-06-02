import numpy as np 

def solveEnv(aMat, chi):
    eigenValuesEnv, eigenVectorsEnv = np.linalg.eigh(aMat)
    # print(eigenValuesEnv)
    if (chi > eigenVectorsEnv.shape[1]):
        return eigenVectorsEnv, 0.0
    else:
        return eigenVectorsEnv[:, -chi:], 1.0 - np.sum(eigenValuesEnv[-chi:]) / np.sum(eigenValuesEnv)
    # return eigenVectorsEnv[:, -chi:]