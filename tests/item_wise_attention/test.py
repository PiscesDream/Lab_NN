import numpy as np
import theano
import theano.tensor as T
from sklearn.cross_validation import train_test_split
from NN.common.data import generate_random_walk

#from NN.RNN.LSTM import stackLSTM
from NN.RNN.attention.itemwise_hard import AttentionModel

def demo():
    x = np.random.uniform(0, 1, size=(100, 5, 3)).astype(theano.config.floatX) # n * T * d_i
    y = np.random.randint(10, size=(100)).astype('int32')

    net = stackLSTM(3, [10, 10], 10) # d_i, d_h, d_o

    loss = net.error(x, y)
    output = net.forward(x)
    print map(lambda x: x.shape, output)
    raise Exception

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf, precision=3)
    theano.config.exception_verbosity='high'
#   demo()

    n_per_class = 1000;  classes = 10
    Time = 4
    item_count = 4
    d_i = 5

    # generate single random walk
    rawx, y = generate_random_walk(classes, n_per_class, d_i, Time, decay=1.00) # (classes*n_per_class, Time, d_i)
    x = np.random.uniform(-100,100, (n_per_class*classes, Time, item_count, d_i)).astype('float32')
    mask = np.random.randint(0, item_count, size=Time)
    for i in xrange(n_per_class*classes):
        for t in xrange(Time):
            x[i, t, mask[t]] = rawx[i, t]
    # x: N * Time spac: * item count * item feature len(1)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.33)
    print mask
    print x.shape
    print y.shape

    net = AttentionModel(
        item_count=item_count,
        dim_input=d_i,
        glimpse_times=Time,
        dim_hidden=20, dim_fc=[], dim_out=classes,
        reward_base=None, 
        activation=T.tanh, bptt_truncate=-1, 
        lmbd=0.5,
        DEBUG=False,)

    net.fit(trainx, trainy, batch_size=2500,
            lr=1e-1,
            max_iter=100000, 
            test_iter=500,      # test on validation set
            disp_iter=10,        # display
            lr_iter=10000,         # update lr
            reward_iter=10,
            decay=0.90,
            gamma=0.50, # gamma*reward + (1-gamma)*rewardi
            val=(testx, testy),
            quick_save=(2000, 'temp.model'),
            )


