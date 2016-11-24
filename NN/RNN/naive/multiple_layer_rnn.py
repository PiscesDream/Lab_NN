import theano 
import theano.tensor as T
import numpy as np
from NN.common.layers import generate_wb 

class RNN(object):
    def __init__(self, x, dim_input, dim_hidden, activation=T.tanh, bptt_truncate=-1, name='RNN'):
        # x is in (N * Time * feature)
        w_input = generate_wb(dim_input, dim_hidden, '{}_input'.format(name), params=['w'])
        w_hidden, b_hidden = generate_wb(dim_hidden, dim_hidden, '{}_hidden'.format(name), params=['w', 'b'])

        def forward(x_t, s_prev, w_i, w_h, b_h): 
            # current input, previous hidden state, w_input, w_hidden, w_output
            # x.shape = n * feature size (scaning on time span)
            s = activation( x_t.dot(w_i) + s_prev.dot(w_h) + b_h )     # n * dim_h
            return s # n*dim_h

        s,updates = theano.scan(
            fn=forward,
            sequences = x.swapaxes(0, 1),
            outputs_info = T.zeros((x.shape[0], dim_hidden)),
            non_sequences = [w_input, w_hidden, b_hidden],
            truncate_gradient=bptt_truncate,
            strict = True)

        self.output = s.swapaxes(0, 1) # N * Time * feature
        self.params = [w_input, w_hidden, b_hidden]

class InputLayer(object):
    def __init__(self, input):
        self.output = input
        self.params = []

class FullConnectLayer(object):
    def __init__(self, input, dim_input, dim_output, activation, name):
        w_output, b_output = generate_wb(dim_input, dim_output, name, params=['w', 'b'])
        self.output = activation(input.dot(w_output) + b_output)
        self.params = [w_output, b_output]

class SoftmaxLayer(object):
    def __init__(self, input):
        self.output = T.nnet.softmax(input)
        self.params = []

class stackRNN(object):
    def __init__(self, shapes, activation=T.tanh, bptt_truncate=4):
        x = T.tensor3('x') # number of sequence * time span * feature size
        y = T.ivector('y')  # only one label (last one) is available
                           # (mask = T.matrix('mask')  # 1 1 1 1 0 0)
        lr = T.scalar('lr', dtype=theano.config.floatX)
    
        layers = [InputLayer(x)]
        for ind, shape in enumerate(zip(shapes[:-2], shapes[1:-1])): # Input, dim1, dim2, ..., dimL, Output
            layer = RNN(layers[-1].output, shape[0], shape[1], activation, bptt_truncate, 'L{}'.format(ind))
            layers.append(layer)

        fc = FullConnectLayer(layers[-1].output[:, -1, :], shapes[-2], shapes[-1], activation, 'FC')
        sm = SoftmaxLayer(fc.output)
        layers.extend([fc, sm])

        output = sm.output 
        loss = T.sum(T.nnet.categorical_crossentropy(output, y))
        prediction = output.argmax(1)

        params = reduce(lambda x, y: x+y.params, layers, []) 
        updates = map(lambda x: (x, x-lr*T.grad(loss, x)), params)

        print 'compile step()'
        self.step = theano.function([x, y, lr], [loss], updates=updates)
#       print 'compile error()'
#       self.error = theano.function([x, y], loss)
#       print 'compile forward()'
#       self.forward = theano.function([x], map(lambda x: x.output, layers))#[layers[-3].output, fc.output])
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
                print 'Iter[{}] lr={}'.format(i, lr)
                print '\tcost: {}'.format(np.mean(cost))
 
