import CTL.funcs.xplib as xplib

def setXP(newXP):
    '''
    Set the *py(e.g. numpy, cupy) library for CTL

    newXP : object, default numpy
        The numpy-like library for numeric functions.
    '''
    xplib.xp = newXP