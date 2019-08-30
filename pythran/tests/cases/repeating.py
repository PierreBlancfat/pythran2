#from: http://stackoverflow.com/questions/14553331/how-to-improve-numpy-performance-in-this-short-code
#pythran export repeating(float[], int)
#runas import numpy as np ; a = np.arange(10, dtype=float); repeating(a, 2)
#bench import numpy as np ; a = np.random.rand(10000); repeating(a, 20)

import numpy as np

def repeating(x, nvar_y):
    nvar_x = x.shape[0]
    y = np.empty(nvar_x*(1+nvar_y))
    y[0:nvar_x] = x[0:nvar_x]
    y[nvar_x:] = np.repeat(x,nvar_y)
    return y
