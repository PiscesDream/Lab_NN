import numpy as np
from sklearn.cross_validation import train_test_split

import sys

import theano
import theano.tensor as T
from NN.RNN.attention.itemwise_hard import AttentionModel

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
    x = x.reshape(x.shape+(1,))
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


    net = AttentionModel(
        item_count=1000,
        dim_input=1,
        glimpse_times=10,
        dim_hidden=200, dim_fc=[], dim_out=10,
        reward_base=None, 
        activation=T.tanh, bptt_truncate=-1, 
        lmbd=0.05,
        DEBUG=False,)

    net.fit(trainx, trainy, batch_size=500,
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



