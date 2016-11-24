from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import numpy as np

import theano
import theano.tensor as T

from NN.FC import stackFC

if __name__ == '__main__':
    theano.config.exception_verbosity='high'

    mnist = fetch_mldata('MNIST original', data_home='~/scikit_learn_data/lfw_home')
    x, y = mnist.data.astype('float32'), mnist.target.astype('int32')
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.33)
    print x.shape
    print y.shape

    net = stackFC(28*28, [256, 256], 10)
    net.fit(trainx, trainy, 
            batch_size=1000,
            lr=1e-3,
            max_iter=2000, 
            test_iter=1000,      # test on validation set
            disp_iter=10,        # display
            lr_iter=2000,         # update lr 
            val=(testx, testy)
            )


