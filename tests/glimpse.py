from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import numpy as np
import theano
import theano.tensor as T

from NN.RNN.attention.glimpse import AttentionModel

if __name__ == '__main__':
    theano.config.exception_verbosity='high'

    mnist = fetch_mldata('MNIST original', data_home='~/scikit_learn_data/lfw_home')
    x, y = mnist.data.astype('float32'), mnist.target.astype('int32')
    a = int(x.shape[1] ** .5)
    x = x.reshape(x.shape[0], a, a)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.10)
    print x.shape
    print y.shape


    # AttentionModel: glimpse_shape, glimpse_times, dim_hidden, dim_out, activation=T.tanh, bptt_truncate=-1
    net = AttentionModel(
        glimpse_shape=(10,10,1), 
        glimpse_times=8, 
        dim_hidden=500, 
        dim_fc=[2000, 500], 
        dim_out=10,
        reward_coef=1.0)

#   d = net.debug(x[:100], y[:100], 0.01)
#   print d
#   print d.shape
#   print net.error(x, y)
#   print map(lambda x: x.shape, net.forward(x))
#   loc = net.locate(x)
#   print 'loc.shape:', loc.shape
#   print loc[:3] 


    net.fit(trainx, trainy, batch_size=5000,
            lr=1e-1,
            max_iter=100000, 
            test_iter=50,      # test on validation set
            disp_iter=5,        # display
            lr_iter=1000,         # update lr
            decay=0.90,
            val=(testx, testy))



