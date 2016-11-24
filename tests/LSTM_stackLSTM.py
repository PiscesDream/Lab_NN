import numpy as np
import theano
from sklearn.cross_validation import train_test_split
from NN.common.data import generate_random_walk
from NN.RNN.LSTM import stackLSTM

def demo():
    x = np.random.uniform(0, 1, size=(100, 5, 3)).astype(theano.config.floatX) # n * T * d_i
    y = np.random.randint(10, size=(100)).astype('int32')

    net = stackLSTM(3, [8, 20], 10) # d_i, d_h, d_o

    loss = net.error(x, y)
    output = net.forward(x)
    print map(lambda x: x.shape, output)
    raise Exception

if __name__ == '__main__':
    theano.config.exception_verbosity='high'
#   demo()

    d_i = 5 
    Time = 4
    n_per_class = 500
    classes = 10

    x, y = generate_random_walk(classes, n_per_class, d_i, Time, decay=0.80)
    print x.shape
    print y.shape
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.33)

    net = stackLSTM(d_i, [20], classes, bptt_truncate=-1)
    net.fit(trainx, trainy, lr=1e-4, 
            max_iter=100000, 
            test_iter=100,       # test on validation set
            disp_iter=10,        # display
            lr_iter=20,        # update lr
            val = (testx, testy)
            )


