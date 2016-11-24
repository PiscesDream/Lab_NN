from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import numpy as np
import theano
import theano.tensor as T

from NN.RNN.attention.glimpse import AttentionModel

# accn = 0.726714285714
#   net = AttentionModel( glimpse_shape=(12,12,1), glimpse_times=8, dim_hidden=256, dim_fc=[], dim_out=10, reward_base=None, activation=T.nnet.sigmoid, rng_std=2.0, lmbd=0.10, # gdupdate + lmbd * rlupdate)
#   net.fit(trainx, trainy, batch_size=1000, lr=5e-1, max_iter=100000, test_iter=50,                  disp_iter=1,       lr_iter=100,       reward_iter=10, decay=0.90, gamma=0.90, val=(testx, testy))

# acc = 5128/7000 = 0.732571428571
#   net = AttentionModel( glimpse_shape=(12,12,1), glimpse_times=8, dim_hidden=512, dim_fc=[], dim_out=10, reward_base=None, activation=ReLU, rng_std=2.0, lmbd=0.10, )
#   net.fit(trainx, trainy, batch_size=5000, lr=5e-1, max_iter=100000, test_iter=50,    disp_iter=1,     lr_iter=100,     reward_iter=10, decay=0.90, gamma=0.90, val=(testx, testy)) 

# log
# temp.model
#   net = AttentionModel( glimpse_shape=(12,12,1), glimpse_times=8, dim_hidden=512, dim_fc=[], dim_out=10, reward_base=None, activation=T.nnet.sigmoid, rng_std=3.0, lmbd=10.00,)
#   net.fit(trainx, trainy, batch_size=2000, lr=5e-1, max_iter=100000, test_iter=50,     disp_iter=1,      lr_iter=100,      reward_iter=10, decay=0.90, gamma=0.20, val=(testx, testy), quick_save=(20, 'temp.model'),)

# log2
# temp2.model 
#   net = AttentionModel( glimpse_shape=(8,8,1), glimpse_times=7, dim_hidden=256, dim_fc=[], dim_out=10, reward_base=None, activation=ReLU, rng_std=2.0, lmbd=5.00,)
#   net.fit(trainx, trainy, batch_size=5000, lr=1e-0, max_iter=100000, test_iter=50,      # test on validation set disp_iter=1,        # display lr_iter=100,         # update lr reward_iter=10, decay=0.90, gamma=0.50, # gamma*reward + (1-gamma)*rewardi val=(testx, testy), quick_save=(20, 'temp2.model'),)

# log3 
# temp3.model
#   net = AttentionModel( glimpse_shape=(8,8,1), glimpse_times=7, dim_hidden=256, dim_fc=[], dim_out=10, reward_base=None, activation=T.nnet.sigmoid, rng_std=3.0, lmbd=10.00,)
#   net.fit(trainx, trainy, batch_size=5000, lr=1e-0, max_iter=100000, test_iter=50,     disp_iter=1,      lr_iter=500,        reward_iter=10, decay=0.90, gamma=0.20, # gamma*reward + (1-gamma)*rewardi val=(testx, testy), quick_save=(20, 'temp2.model'),)

# log4
# temp2.model
#   net = AttentionModel( glimpse_shape=(8,8,1), glimpse_times=7, dim_hidden=256, dim_fc=[], dim_out=10, reward_base=None, activation=T.nnet.sigmoid, rng_std=3.0, lmbd=10.00,)
#   net.fit(trainx, trainy, batch_size=5000, lr=1e-0, max_iter=100000, test_iter=50,     disp_iter=1,      lr_iter=100,        reward_iter=10, decay=0.30, gamma=0.20, # gamma*reward + (1-gamma)*rewardi val=(testx, testy), quick_save=(20, 'temp2.model'),)

def ReLU(x):
    return T.maximum(x, 0)

def test():
    net = AttentionModel(
        glimpse_shape=(8,8,1),
        glimpse_times=7,
        dim_hidden=256,
        dim_fc=[], 
        dim_out=10,
        reward_base=None,
        activation=ReLU,
        rng_std=2.0,
        lmbd=5.00, # gdupdate + lmbd * rlupdate
        )

    net.fit(trainx, trainy, batch_size=50,
            lr=1e-0,
            max_iter=100000, 
            test_iter=50,      # test on validation set
            disp_iter=1,        # display
            lr_iter=100,         # update lr
            reward_iter=10,
            decay=0.90,
            gamma=0.50, # gamma*reward + (1-gamma)*rewardi
            val=(testx, testy),
#           quick_save=(20, 'temp2.model'),
            )

def train():
    net = AttentionModel( 
        glimpse_shape=(8,8,1), 
        glimpse_times=7, 
        dim_hidden=256, 
        dim_fc=[], 
        dim_out=10, 
        reward_base=None, 
        activation=T.nnet.sigmoid, 
        rng_std=3.0, 
        lmbd=10.00,)

    net.fit(trainx, trainy, 
        batch_size=5000, 
        lr=1e-0, 
        max_iter=100000, 
        test_iter=50,     
        disp_iter=1,      
        lr_iter=500,        
        reward_iter=10, 
        decay=0.90, 
        gamma=0.20, # gamma*reward + (1-gamma)*rewardi 
        val=(testx, testy), 
        quick_save=(20, 'temp3.model'),
    )

def debug():
    net = AttentionModel(
        glimpse_shape=(12,12,1),
        glimpse_times=8,
        dim_hidden=256,
        dim_fc=[], 
        dim_out=10,
        reward_base=None,
        activation=ReLU,
        rng_std=3.0,
        lmbd=0.10, # gdupdate + lmbd * rlupdate
        )
    
    reward = np.zeros((8)).astype('float32')
    d = net.debug(x[:1], y[:1], 0.01, reward)
    print net.error(x, y, reward)
    print map(lambda x: x.shape, net.forward(x[:1]))
    loc_mean, location = net.locate(x[:1]) 
    print 'loc mean:', loc_mean
    raise Exception

if __name__ == '__main__':
    theano.config.exception_verbosity='high'

    mnist = fetch_mldata('MNIST original', data_home='~/scikit_learn_data/lfw_home')
    x, y = mnist.data.astype('float32'), mnist.target.astype('int32')
    a = int(x.shape[1] ** .5)
    x = x.reshape(x.shape[0], a, a)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.10)
    print x.shape
    print y.shape

#   debug()
#   test()
#   train()


    net = AttentionModel( glimpse_shape=(8,8,1), glimpse_times=7, dim_hidden=256, dim_fc=[], dim_out=10, reward_base=None, activation=T.nnet.sigmoid, rng_std=3.0, lmbd=10.00,)
