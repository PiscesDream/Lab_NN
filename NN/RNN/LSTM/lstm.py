# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import numpy  as np
import sys

from NN.common.layers import generate_wb 
from NN.common.layers import InputLayer, SoftmaxLayer, FullConnectLayer

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

class LSTM(object):
    def __init__(self, x, dim_input, dim_hidden, activation=T.nnet.sigmoid, bptt_truncate=-1, name='LSTM'):
        '''
            f_t =       g(      W_xf·x_t + W_hf·h_prev + b_f )
            i_t =       g(      W_xi·x_t + W_hi·h_prev + b_i )
            C_t_hat =   tanh(   W_xc·x_t + W_hc·h_prev + b_c )
            o_t =       g(      W_xo·x_t + W_ho·h_prev + b_o )
            C_t =       f_t*C_prev + i_t*C_t_hat
            h_t =       o_t * tanh(C_t)
        '''
        W_x = generate_wb(dim_input, 4*dim_hidden, '{}_x'.format(name), params=['w'])
        W_h, b_h = generate_wb(dim_hidden, 4*dim_hidden, '{}_hidden'.format(name))

        def _slice(_x, n, dim):
            if _x.ndim == 3: return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def step(x_t, h_prev, C_prev, W_x, W_h, b_h):
            # N*4dh = N*di·di*4dh   +  N*dh·dh*4dh          + 1*4dh 
            res    =   x_t.dot(W_x) +  h_prev.dot(W_h)      + b_h.dimshuffle('x', 0) 
            
            f = activation(res[:, 0*dim_hidden:1*dim_hidden]) # N * dh
            i = activation(res[:, 1*dim_hidden:2*dim_hidden]) # N * dh
            C_hat = T.tanh(res[:, 2*dim_hidden:3*dim_hidden]) # N * dh
            o = activation(res[:, 3*dim_hidden:4*dim_hidden]) # N * dh

            C = f*C_prev + i*C_hat # N * dh
            h = o * T.tanh(C)      # N * dh
            return h, C

        [h, C], updates = theano.scan(\
            fn = step,
            sequences = x.swapaxes(0, 1), # N*Time*feature->Time*N*feature
            outputs_info = [T.zeros((x.shape[0], dim_hidden), theano.config.floatX),
                            T.zeros((x.shape[0], dim_hidden), theano.config.floatX)],
            non_sequences = [W_x, W_h, b_h], 
            truncate_gradient = bptt_truncate,
            strict = True)
        
        self.output = h.swapaxes(0, 1) # Time*N*hidden-> N*Time*hidden
        self.params = [W_x, W_h, b_h]

class stackLSTM(object):
    def __init__(self, dim_input, shapes, dim_output, activation=T.tanh, bptt_truncate=-1):
        x = T.tensor3('x') # number of sequence * time span * feature size
        y = T.ivector('y')  # only one label (last one) is available
        lr = T.scalar('lr', dtype=theano.config.floatX)
    
        layers = [InputLayer(x)]
        shapes = [dim_input] + shapes
        for ind, shape in enumerate(zip(shapes[:-1], shapes[1:])): # Input, dim1, dim2, ..., dimL, Output
            layer = LSTM(layers[-1].output, shape[0], shape[1], activation, bptt_truncate, 'LSTM[{}]'.format(ind))
            layers.append(layer)

        fc = FullConnectLayer(layers[-1].output[:, -1, :], shapes[-1], dim_output, activation, 'FC')
        sm = SoftmaxLayer(fc.output)
        layers.extend([fc, sm])

        output = sm.output
        loss = T.sum(T.nnet.categorical_crossentropy(output, y))
        prediction = output.argmax(1)

        params = reduce(lambda x, y: x+y.params, layers, [])
        updates = map(lambda x: (x, x-lr*T.grad(loss, x)), params)

        print 'compile step()'
        self.step = theano.function([x, y, lr], [loss], updates=updates)
        print 'compile error()'
        self.error = theano.function([x, y], loss)
        print 'compile forward()'
        self.forward = theano.function([x], map(lambda x: x.output, layers))#[layers[-3].output, fc.output])
        print 'compile predict()'
        self.predict = theano.function([x], prediction)

    def fit(self, x, y, lr=1e-2,
            max_iter=100000, 
            test_iter=100,      # test on validation set
            disp_iter=1,        # display
            lr_iter=10,         # update lr
            val=None):
        valx, valy = val if val !=None else (x, y)

        lastcost = np.inf
        cost = []
        for i in xrange(max_iter):
            if i%test_iter == 0:
                correct = (self.predict(valx) == valy).sum()
                print '\n\tacc = {}/{} = {}'.format(correct, len(valy), float(correct)/len(valy))
            if i % lr_iter == 0:
                cost = np.mean(cost)
                lr = lr * 0.8 if cost >= lastcost else lr * 1.02
                lastcost = cost
                cost = []

            # update
            cost.append( self.step(x, y, lr) )

            if i % disp_iter == 0: 
                sys.stdout.flush()
                print 'Iter[{}] lr={}'.format(i, lr)
                print '\tcost: {}'.format(np.mean(cost))
 

