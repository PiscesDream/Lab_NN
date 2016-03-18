# not finish yet
import theano
import theano.tensor as T
import numpy as np


rng = np.random.RandomState(32)
srng = T.shared_randomstreams.RandomStreams(rng.randint(100000000))

class DropoutLayer(object):
    def __init__(self, input, shape, p, active=T.nnet.sigmoid, test_input=None):
        self.input = input
        if test_input == None: test_input = input

        li, lo = shape #self.shape = init
        init = np.random.uniform(-6/np.sqrt(li+lo), 6/np.sqrt(li+lo), size=shape)

        self.w = theano.shared(name='w', value=init.astype(theano.config.floatX)) # I * O
        self.b = theano.shared(name='b', value=np.zeros((lo), dtype=theano.config.floatX) ) # 1 * O
        self.params = [self.w, self.b]

        mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
        self.output = active((mask*input).dot(self.w) + self.b) # N * O
        self.test_output = active(test_input.dot(self.w*p) + self.b*p) # N * O

class InputLayer(object):
    def __init__(self, input, test_input=None):
        self.input = input
        self.output = input
        self.test_output = input if test_input == None else test_input
        self.params = []

class SoftmaxLayer(object):
    def __init__(self, input, test_input=None):
        self.input = input
        self.output = T.nnet.softmax(input)
        self.test_output = self.output if test_input == None else T.nnet.softmax(test_input)
        self.params = []

class Net(object):
    def __init__(self, layer_shapes, dropoutps):
        assert (len(layer_shapes)-1 == len(dropoutps)) # dropout layers + softmax layer
        layer_shapes = zip(layer_shapes[:-1], layer_shapes[1:])

        x = T.matrix('x', dtype='float32')
        y = T.ivector('y')
        lr = T.scalar('lr', dtype='float32')

        self.layers = [InputLayer(x)]
        for shape, dropoutp in zip(layer_shapes, dropoutps):
            layer = DropoutLayer(self.layers[-1].output, shape, dropoutp, test_input=self.layers[-1].test_output)
            self.layers.append(layer)
        self.layers.append(SoftmaxLayer(self.layers[-1].output, test_input=self.layers[-1].test_output))

        self.output = self.layers[-1].output
        neg_likelihood = -T.log(self.output[T.arange(y.shape[0]), y]).sum() #y without boarden
        cost = neg_likelihood 

        self.params = [] 
        update = []
        for layer in self.layers:
            self.params.extend(layer.params)
        for param in self.params:
            update.append( (param, T.cast(param-lr*T.grad(cost, param), 'float32')  ))
        
        self.update = update
        self.x = x
        self.y = y
        self.lr = lr
        self.cost = cost

        self.test_output = self.layers[-1].test_output
        self.predict = theano.function([x], self.test_output.argmax(1))

    def fit(self, x, y, lr=1e-0, 
            max_iter=100000, 
            test_iter=100,      # test on validation set
            disp_iter=1,        # display
            lr_iter=10,         # update lr
            val=None):
        valx, valy = val if val !=None else (x, y)
        step = theano.function([self.lr], self.cost, 
            givens={self.x:x, self.y:y},
            updates=self.update) # use given to xxx

        lastcost = np.inf
        cost = []
        for i in xrange(max_iter):
            if i%test_iter == 0:
                prediction = self.predict(valx)
#               print zip(prediction, valy)
                correct = (prediction == valy).sum()
                print '\n\tacc = {}/{} = {}'.format(correct, len(valy), float(correct)/len(valy))
            if i % lr_iter == 0:
                cost = np.mean(cost)
                lr = lr * 0.5 if cost > lastcost else lr * 1.01
                lastcost = cost
                cost = []

            # update
            cost.append( step(lr) )

            if i % disp_iter == 0: 
                print 'Iter[{}] lr={}'.format(i, lr)
                print '\tcost: {}'.format(np.mean(cost))
        
if __name__ == '__main__':
    theano.config.exception_verbosity = 'high'

#   from sklearn.datasets import load_iris
#   iris = load_iris()
#   x, y = iris.data.astype(theano.config.floatX), iris.target.astype('int32')
#   trainx = valx = x
#   trainy = valy = y
#   print y
#   net = Net([4, 40, 3], [0.8, 0.4])
#   net.fit(trainx, trainy, lr=1e-0, test_iter=1000, disp_iter=100, lr_iter=10000, val=(valx, valy))

    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home='~/scikit_learn_data/lfw_home')
    x = mnist.data.astype(theano.config.floatX)
    y = mnist.target.astype('int32')

    index = np.random.permutation(len(y))
    x = x[index]
    y = y[index]

    trainx = x[:50000]
    trainy = y[:50000]
    valx = x[50000:60000]
    valy = y[50000:60000]

    net = Net([784, 50, 50, 10], [0.8, 0.6, 0.4])
    net.fit(trainx, trainy, lr=1e-0, lr_iter=10, test_iter=10, val=(valx, valy))
    print net.predict(x)





