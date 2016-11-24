import theano
import theano.tensor as T
import numpy as np

class InputLayer(object):
    def __init__(self, input):
        self.output = input
        self.params = []

class SoftmaxLayer(object):
    def __init__(self, input):
        self.output = T.nnet.softmax(input)
        self.params = []

class FullConnectLayer(object):
    def __init__(self, input, dim_input, dim_output, activation=T.tanh, name='', givens={}):
        w_output, b_output = generate_wb(dim_input, dim_output, name, params=['w', 'b'], givens=givens)
        self.output = activation(input.dot(w_output) + b_output)
        self.params = [w_output, b_output]

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)

def generate_wb(dim1, dim2, label, params=['w', 'b'], givens={}): # generate a set of w and b for transformation from dim1 to dim2
    upper = 6/np.sqrt(dim1+dim2)
    lower = -upper 
    rets = []
    if 'w' in params:
        if 'w' in givens:
            w = givens['w']
        else:
            w = theano.shared(name='{}->w'.format(label),
                          value=np.random.uniform(upper, lower, size=(dim1, dim2)).astype(theano.config.floatX))
        rets.append(w)
    if 'b' in params:
        if 'b' in givens:
            b = givens['b']
        else:
            b = theano.shared(name='{}->b'.format(label),
                          value=np.random.uniform(upper, lower, size=(dim2)).astype(theano.config.floatX))
        rets.append(b)
    return rets[0] if len(rets) == 1 else rets

