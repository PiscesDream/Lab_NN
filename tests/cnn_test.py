from sklearn.datasets import fetch_mldata
from NN.CNN import stackCNN
from sklearn.cross_validation import train_test_split

import theano
import theano.tensor as T

if __name__ == '__main__':
    theano.config.exception_verbosity='high'

    mnist = fetch_mldata('MNIST original', data_home='~/scikit_learn_data/lfw_home')
    x, y = mnist.data.astype('float32'), mnist.target.astype('int32')
    a = int(x.shape[1] ** .5)
    x = x.reshape(x.shape[0], 1, a, a)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.33)
    print x.shape
    print y.shape

    net = stackCNN(dim_in=(1, a, a), 
                   dim_filters=[(20, 3, 3), (10, 2, 2)],
                   dim_fc=[300, 10],
                   activation=T.nnet.sigmoid)

    net.fit(trainx, trainy,
            batch_size=500,
            lr=5e-1,
            max_iter=100000, 
            test_iter=1000,      # test on validation set
            disp_iter=10,        # display
            lr_iter=100,         # update lr)
            val=(testx, testy))
            

