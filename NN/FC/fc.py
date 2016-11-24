import theano
import theano.tensor as T
import numpy as np

from NN.common.layers import InputLayer, SoftmaxLayer, FullConnectLayer
from NN.common.nets import NetModel

class stackFC(NetModel):
    def __init__(self, dim_in, dim_hidden, dim_out, activation=T.tanh):
        x = T.fmatrix('x')
        y = T.ivector('y')
        lr = T.fscalar('lr')

        layers = [InputLayer(x)]
        for ind, (Idim, Odim) in enumerate(zip([dim_in]+dim_hidden, dim_hidden+[dim_out])):
            fc = FullConnectLayer(layers[-1].output, Idim, Odim, activation=activation,\
                                        name='FC[{}]'.format(ind))
            layers.append(fc)
        sm = SoftmaxLayer(layers[-1].output)
        layers.append(sm)

        output = sm.output 
        logloss = T.nnet.categorical_crossentropy(T.clip(output, 1e-15, 1-1e-15), y)
        loss = T.sum(logloss)/y.shape[0]
        prediction = output.argmax(1)

        params = reduce(lambda x, y: x+y.params, layers, []) 
        updates = map(lambda x: (x, x-lr*T.grad(loss, x)), params)

        print 'compile step()'
        self.step = theano.function([x, y, lr], [loss], updates=updates)
        print 'compile predict()'
        self.predict = theano.function([x], prediction)
        print 'compile predict_proba()'
        self.predict_proba = theano.function([x], output)

        # for saving
        self.params = params 


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
                print '\n\ttest acc = {}/{} = {}'.format(correct, len(valy), float(correct)/len(valy))
                correct = (self.predict(x) == y).sum()
                print '\n\ttrain acc = {}/{} = {}'.format(correct, len(y), float(correct)/len(y))
            if i % lr_iter == 0:
                if np.mean(cost) >= lastcost:
                    lr *= 0.5
                else:
                    lr *= 1.01
                lastcost = np.mean(cost)
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
 

