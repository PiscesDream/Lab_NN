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
    def __init__(self, input, dim_input, dim_output, activation=T.tanh, name=''):
        w_output, b_output = generate_wb(dim_input, dim_output, name, params=['w', 'b'])
        self.output = activation(input.dot(w_output) + b_output)
        self.params = [w_output, b_output]

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

