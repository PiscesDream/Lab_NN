import numpy as np
from sklearn.cross_validation import train_test_split

import sys

import theano
import theano.tensor as T

from NN.RNN.LSTM import stackLSTM

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf, precision=3)
    theano.config.exception_verbosity='high'

    K = 500
    Time = 1.0
    G = 10
    sys.path.append('/home/shaofan/Projects/MMAPM/')
    import names
    print names.dataset_name 
    data = np.load(open(names.histogramFilename(K, G), 'r'))
    x, y = data['x'], data['y'] 
    try:
        train_test = data['train_test']
        trainx, trainy = x[train_test==2],  y[train_test==2]
        valx, valy = x[train_test==1], y[train_test==1]
        testx, testy = x[train_test==0], y[train_test==0]
    except:
        trainx, testx, trainy, testy = \
            train_test_split(x, y, test_size=0.32, random_state=3)
        valx, valy = testx, testy
    print trainx.shape
    print trainy.shape
    print testx.shape
    print testy.shape
    print np.bincount(trainy)

    net = stackLSTM(1000, [750, 500], 10, bptt_truncate=-1)
    net.fit(trainx, trainy, lr=1e-4, 
            max_iter=10000000, 
            test_iter=100,       # test on validation set
            disp_iter=10,        # display
            lr_iter=20,        # update lr
            val = (testx, testy)
            )


