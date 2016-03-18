import numpy as np
from numpy import linalg

def PSD(mat):
    mat = (mat+mat.T)/2.0
    eig, eigv = linalg.eig(mat)
    eig = np.maximum(eig, 0)
    eig = np.diag(eig)
    mat = eigv.dot(eig).dot(eigv.T) 
    return mat


import theano
import theano.tensor as T

def generate_wb(dim1, dim2, label, params=['w', 'b']): # generate a set of w and b for transformation from dim1 to dim2
    upper = 6/np.sqrt(dim1+dim2)
    lower = -upper 
    rets = []
    if 'w' in params:
        w = theano.shared(name='w_{}'.format(label),
                          value=np.random.uniform(upper, lower, size=(dim1, dim2)).astype(theano.config.floatX))
        rets.append(w)
    if 'b' in params:
        b = theano.shared(name='b_{}'.format(label),
                          value=np.random.uniform(upper, lower, size=(dim2)).astype(theano.config.floatX))
        rets.append(b)
    return rets[0] if len(rets) == 1 else rets


