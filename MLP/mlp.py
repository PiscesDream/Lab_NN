'''
    140819:
        move lr into fit
        add time recording
        add pred
        add prob
    2014/09/03:
        change the last layer into softmax
    2015/03/26:
        add the ReLU layer
        fix the no cnn bug
'''
import cPickle
import gzip
import os
import sys
import time

import numpy
from numpy import *

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv, softmax

import time

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

        print activation

class conv_pool_layer(object):
    
    def __init__(self, rng, input, filter_shape, image_shape, W = None, b = None, poolsize=(2, 2),
                activation=T.tanh):
        
        if W is None:
            #calc the W_bound and init the W
            fan_in = numpy.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                       numpy.prod(poolsize))
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype = theano.config.floatX),
                            borrow = True)
        else:
            self.W = W

        if b is None:
            b_value = numpy.zeros((filter_shape[0],), 
                                   dtype = theano.config.floatX)
            self.b = theano.shared(value = b_value, borrow = True)
        else:
            self.b = b


        conv_out = conv.conv2d(input = input, filters = self.W, 
                filter_shape = filter_shape, image_shape = image_shape)
        pooled_out = downsample.max_pool_2d(input = conv_out,
                                ds = poolsize, ignore_border = True)

        self.output = activation(pooled_out + self.b.dimshuffle('x',0,'x','x'))

        self.params = [self.W, self.b]

def raw_dump(cnn, filename):
    import cPickle
    params = []
    for ele in cnn.cnn_layers+cnn.hid_layers:
        params.append(ele.W.get_value())
        params.append(ele.b.get_value())
    cPickle.dump((cnn.dim_in, cnn.size_in, cnn.size_out, cnn.lmbd, cnn.nkerns, cnn.nhiddens[:-1], cnn.time, params), open(filename, 'wb'))

def raw_load(filename):
    import cPickle
    dim_in, size_in, size_out, lmbd, nkerns, nhiddens, time, params = cPickle.load(open(filename, 'rb'))
    ncnn = cnn(dim_in, size_in, size_out, lmbd, nkerns, nhiddens)
    ncnn.time = time
    ind = 0
#    print len(params)
#    print len(ncnn.cnn_layers+ncnn.hid_layers)
#    print ncnn.hid_layers
#    print ncnn.cnn_layers
    for ele in ncnn.cnn_layers+ncnn.hid_layers:
        ele.W.set_value(params[ind])
        ele.b.set_value(params[ind+1])
        ind += 2
    return ncnn

class cnn(object):

    def __init__(self, dim_in, size_in, size_out, lmbd = 0.01, 
                 nkerns = [(20, (8, 8), (2, 2))], nhiddens = [10],
                 activation=T.tanh): 
                 
        x = T.tensor4('x')
        y = T.ivector('y')
        lr = T.dscalar('lr')
        i_shp = array(size_in)
        rng = random.RandomState(10)

        #building model
        print '..building the model'

        #building the cnn
        cnn_layers = []
        if nkerns!=[]:
            for ind, ele in enumerate(nkerns):
                if ind == 0:#can opt here
                    input = x
                    filter_shape = (ele[0], dim_in)+ele[1]
                else:
                    input = cnn_layers[-1].output
                    filter_shape = (ele[0], nkerns[ind-1][0]) + ele[1]

                layer = conv_pool_layer(rng,
                                        input = input,
                                        image_shape = None,
                                        filter_shape = filter_shape ,
                                        poolsize = ele[2],
                                        activation = activation) 

                cnn_layers.append(layer)
                i_shp = (i_shp - array(ele[1]) + 1)/array(ele[2])
        #raw_input('pause')
        
        #building the hidden
        hid_layers = []
        nhiddens = nhiddens + [size_out]
        L1 = .0
        L2 = .0
        for ind, ele in enumerate(nhiddens):
            if ind == 0:#can opt here
                input = T.flatten(cnn_layers[-1].output if nkerns!=[] else x, 2)
                n_in = nkerns[-1][0] * i_shp.prod() if nkerns!=[] else size_in[0]*size_in[1]*dim_in
            else:
                input = hid_layers[-1].output
                n_in = nhiddens[ind-1]
            if ind == len(nhiddens)-1:
                this_activation = T.nnet.softmax
            else:
                this_activation = activation #T.nnet.sigmoid

            layer = HiddenLayer(rng, input = input ,
                                n_in = n_in,
                                n_out = ele, 
                                activation = this_activation) 
            L1 += abs(layer.W).sum()
            L2 += (layer.W ** 2).sum()
            hid_layers.append(layer)

        #build cost
#        diff = ((y - hid_layers[-1].output) ** 2).sum()
#        cost = diff + L2 * lmbd
        #log                   T.sum or T.mean is judged
        negative_likelihood = -T.sum(T.log(hid_layers[-1].output)[T.arange(y.shape[0]), y]) #y without boarden
        cost = negative_likelihood + L2 * lmbd

#        errors = T.mean(T.neq(T.argmax(y, 1), T.argmax(hid_layers[-1].output, 1)))
        y_pred = T.argmax(hid_layers[-1].output, axis = 1) 
        errors = T.mean(T.neq(y_pred, y))

        #build update
        params = []
        for ind, ele in enumerate(cnn_layers + hid_layers):
            params.extend(ele.params)
        grads = T.grad(cost, params)

        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - lr * grad_i))

        #setting self
        self.x = x
        self.y = y
        self.cost = cost
        self.errors = errors 
        self.updates = updates
        self.cnn_layers = cnn_layers
        self.hid_layers = hid_layers
        self.nkerns = nkerns
        self.nhiddens = nhiddens
        self.lr = lr 
        self.time = [] 

        self.dim_in = dim_in
        self.size_in = size_in
        self.size_out = size_out
        self.lmbd = lmbd

    def test_error(self, x, y):
        return theano.function([], self.errors,
                               givens={self.x : x, self.y : y})();

    def fit(self, datasets, batch_size = 500, n_epochs = 200, learning_rate = 0.01):
        index = T.lscalar()

        train_set_x, train_set_y= datasets[0]
        test_set_x, test_set_y= datasets[1]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_test_batches /= batch_size

        train_model = theano.function([index], self.cost, 
            updates = self.updates,
            givens = {
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size],
                self.lr: learning_rate})
         
        test_model = theano.function([], self.errors,
            givens = {
                self.x: test_set_x,
                self.y: test_set_y})

        debug_f = theano.function([index], self.errors,
            givens = {
                self.x: test_set_x[index * batch_size : (index+1) * batch_size],
                self.y: test_set_y[index * batch_size : (index+1) * batch_size]})

#        print numpy.mean([debug_f(i) for i in xrange(n_test_batches)]) 
        raw_input(test_model())

        print '...training'
        maxiter = n_epochs
        iteration = 0
        while iteration < maxiter:
            start_time = time.time()
            iteration += 1
            print 'iteration %d' % iteration
            for minibatch_index in xrange(n_train_batches):
                print '\tL of (%03d/%03d) = %f\r' % (minibatch_index, n_train_batches, train_model(minibatch_index)),
            print ''
            print 'error = %f' % test_model()
            self.time.append(time.time()-start_time)

    def __repr__(self):
        return '<CNN: %r; HID: %r>' % (self.nkerns, self.nhiddens)

    def pred(self, x):
        return theano.function([], T.argmax(self.hid_layers[-1].output, 1), 
                        givens = {self.x: x})()

    def prob(self, x):
        return theano.function([], self.hid_layers[-1].output,
                        givens = {self.x: x})() 
    
def boarden(size, y):
    new = zeros((y.shape[0], size))
    for ind, ele in enumerate(y):
        new[ind][ele] = 1
    return new
        

def load_data(dataset, num = None):
    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    train_set = (concatenate([train_set[0], valid_set[0]], 0), concatenate([train_set[1], valid_set[1]], 0))
    f.close()

    def shared_dataset(data_xy, borrow=True, num = None):
        data_x, data_y = data_xy
        if num:
            data_x = data_x[:num]
            data_y = data_y[:num]
#        data_y = boarden(10, data_y)

        size = int(data_x.shape[1]**.5)
        data_x = data_x.reshape(data_x.shape[0], 1, size, size) 
        print data_x.shape, data_y.shape

        shared_x = theano.shared(asarray(data_x,
                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(asarray(data_y,
                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set, num = num)
#    valid_set_x, valid_set_y = shared_dataset(valid_set, num = num)
    train_set_x, train_set_y = shared_dataset(train_set, num = num)

    rval = [(train_set_x, train_set_y), #(valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    theano.config.on_unused_input='ignore'
    size = 28

    datasets = load_data('../../Data/mnist/mnist.pkl.gz')

    transfer_test = False
    if transfer_test:
        cnn_cl = cPickle.load(open('classifier/mnist_cnn_00114.dat', 'rb'))

    predict_and_probability = False
    if predict_and_probability:
        print 'loading Model...'
        k = cPickle.load(open('classifier/mnist_cnn_00114.dat', 'rb'))

        print k

        f = gzip.open('../../Data/mnist/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        x = asarray(train_set[0][:10].reshape(-1, 1, 28, 28), dtype=theano.config.floatX)
        print x.shape
        print k.prob(theano.shared(x))
        print k.pred(theano.shared(x))

    continue_training = False
    if continue_training:
        print 'loading Model...'
#        k = cPickle.load(open('classifier/mnist_cnn_00114.dat', 'rb'))
        k = raw_load('classifier/mnist_cnn_raw_00113.dat')

        print k
        print k.time
        print 'continue training'
        
        k.fit(datasets, n_epochs = 3, batch_size = 100, learning_rate = 0.001)
        raw_dump(k, 'mnist_cnn_raw.dat')
    

    test_loaded_model = False 
    if test_loaded_model:
        print 'loading Model...'
        k = cPickle.load(open('mnist_cnn_classifier_0016.dat', 'rb'))
        print 'testing...'
        print k.test_error(*datasets[0])

    complex_cnn = False 
    if complex_cnn: 
        k = cnn(dim_in = 1, size_in = (28, 28), size_out = 10, 
               nkerns = [(10, (5, 5), (2, 2)), (16, (5, 5), (2, 2))], 
               nhiddens=[90, 30])
        print k
        k.fit(datasets, n_epochs = 40, batch_size = 100, learning_rate = 0.003)

        cPickle.dump(k, open('mnist_cnn_classifier.dat', 'wb'))
        raw_dump(k, 'mnist_cnn_raw.dat')

    simple_cnn = True 
    if simple_cnn:

        def ReLU(x):
            return T.max([x, T.zeros_like(x)], 0)
        ReLU = lambda x: T.max([x, T.zeros_like(x)], 0)
        

        k = cnn(dim_in = 1, size_in = (28, 28), size_out = 10, 
               nkerns = [], #[(10, (3, 3), (2, 2))], 
               nhiddens=[3,3,3,3],
               activation=ReLU)#T.nnet.sigmoid)
        k.fit(datasets, n_epochs = 100, batch_size = 100, learning_rate = 0.001)

        raw_dump(k, 'mnist_cnn_raw.dat')

    test_raw = False 
    if test_raw:
        f = gzip.open('../../Data/mnist/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        x = asarray(train_set[0][:10].reshape(-1, 1, 28, 28), dtype=theano.config.floatX)
        print x[0]
        print x.shape

        new = raw_load('mnist_cnn_raw.dat')
        print new.prob(theano.shared(x))
        print new.pred(theano.shared(x))


