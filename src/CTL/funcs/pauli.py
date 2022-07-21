import CTL.funcs.xplib as xplib

def sigmaX(dtype = None):
    if dtype is None:
        dtype = xplib.xp.complex128
    return xplib.xp.array([
        [0, 1], 
        [1, 0]
    ], dtype = dtype)

def sigmaY(dtype = None):
    if dtype is None:
        dtype = xplib.xp.complex128
    return xplib.xp.array([
        [0, -1j], 
        [1j, 0]
    ], dtype = dtype)

def sigmaZ(dtype = None):
    if dtype is None:
        dtype = xplib.xp.complex128
    return xplib.xp.array([
        [1, 0], 
        [0, -1]
    ], dtype = dtype)

def identity(dtype = None):
    if dtype is None:
        dtype = xplib.xp.complex128
    return xplib.xp.array([
        [1, 0], 
        [0, 1]
    ], dtype = dtype)