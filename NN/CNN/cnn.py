import theano
import theano.tensor as T
import numpy as np
from NN.common.nets import InputLayer, SoftmaxLayer, FullConnectLayer

class ConvolutionLayer(object):
    def __init__(self, input, filter_shape, image_shape, poolsize=(2, 2), activation=T.tanh, W=None, b=None, name=''):
        if W is None:
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:])/np.prod(poolsize))
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(
                np.random.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype = theano.config.floatX), 
                name='<CNN: {}>_W'.format(name),
                borrow = True)
        else:
            self.W = W

        if b is None:
            b_value = np.zeros((filter_shape[0],), dtype = theano.config.floatX)
            self.b = theano.shared(value = b_value, name='<CNN: {}>_b'.format(name), borrow = True)
        else:
            self.b = b

        conv_out = T.nnet.conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
        pooled_out = T.signal.downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)

        self.output = activation(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W, self.b]

class stackCNN(object):
    def __init__(self, dim_in, dim_filters, dim_fc, activation=T.tanh):
        dim_filters_ = dim_filters
        dim_filters = [] 
        for ind, df in enumerate(dim_filters_):
            dim_filters.append( (df[0], dim_in[0] if ind == 0 else dim_filters[-1][0], df[1], df[2]) )
        print dim_filters


        x = T.ftensor4('x')
        y = T.ivector('y')
        lr = T.fscalar('lr')

        layers = [InputLayer(x)]
        image_shape = np.array(dim_in) # channel * W * H
        for dim in dim_filters: # CNN CNN CNN ... FC SOFTMAX 
            layer = ConvolutionLayer(layers[-1].output, filter_shape=dim, image_shape=None, activation=activation)
            layers.append(layer)

            print 'Image shape:', image_shape
            assert image_shape[0] == dim[1] 
            # update shape
            image_shape[0] = dim[0]            
            image_shape[1] = (image_shape[1]-dim[2]+1)/2
            image_shape[2] = (image_shape[2]-dim[3]+1)/2
        print 'Image shape:', image_shape

        layers.append(InputLayer(layers[-1].output.flatten(2)))
        dim_fc = [np.prod(image_shape)] + dim_fc
        for Idim, Odim in zip(dim_fc[:-1], dim_fc[1:]): # CNN CNN CNN ... FC SOFTMAX 
            layer = FullConnectLayer(layers[-1].output, Idim, Odim, activation=activation)
            layers.append(layer)
        sm = SoftmaxLayer(layers[-1].output)
        layers.append(sm)

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
#       self.forward = theano.function([x], map(lambda x: x.output, layers))
        print 'compile predict()'
        self.predict = theano.function([x], prediction)

    def fit(self, x, y, 
            batch_size,
            lr=1e-2,
            max_iter=100000, 
            test_iter=1000,      # test on validation set
            disp_iter=10,        # display
            lr_iter=100,         # update lr
            val=None):
        valx, valy = val if val !=None else (x, y)

        batch_count = len(x)/batch_size

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
            batch_index = np.random.randint(batch_count)
            start = batch_index*batch_size
            end = (batch_index+1)*batch_size
            cost.append( self.step(x[start:end], 
                                   y[start:end], lr) )

            if i % disp_iter == 0: 
                print 'Iter[{}] lr={}'.format(i, lr)
                print '\tcost: {}'.format(np.mean(cost))
 
