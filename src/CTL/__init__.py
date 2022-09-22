import CTL.funcs.xplib as xplib
import CTL.funcs.globalFlag as globalFlag

def setXP(newXP):
    '''
    Set the *py(e.g. numpy, cupy) library for CTL

    newXP : object, default numpy
        The numpy-like library for numeric functions.
    '''
    xplib.xp = newXP

def turnOnOutProductWarning():
    globalFlag.outProductWarning = True

def turnOffOutProductWarning():
    globalFlag.outProductWarning = False

from CTL.tensor.tensor import Tensor