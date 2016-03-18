from preprocess import load_data
#import pyximport; pyximport.install()
#from npmodel import npRNN
from Tmodel import TheanoRNN as RNN
from Tmodel import TheanoGRU as GRU
import sys

if __name__ == '__main__':
    word_dim = 3600 #10000
    hidden_dim = 500 #1000
    sc = -1

    #x, y, dictionary= load_data(filename='../data/reddit-comments-2015.csv', vocabulary_size = word_dim, sentences_count = sc)
    x, y, dictionary= load_data(filename='../data/Sonnets.txt', vocabulary_size = word_dim, sentences_count = sc)
    sys.stdout.flush()
    
    rnn = GRU(word_dim, hidden_dim=hidden_dim, dictionary=dictionary)
    rnn.load('./models/rnn-theano-500-3600-2016-02-04.model')
    rnn.fit(x, y, nepoch=100, nstep=2300, evaluate_loss_after=1, learning_rate=0.000600)

#   o = rnn.forward(x[0])
#   print o


#    rnn.gradient_check([0, 1, 2, 3], [1, 2, 3, 4])

#   print "Expected Loss for random predictions: %f" % np.log(word_dim)
#   print "Actual loss: %f" % rnn.loss(x[:1000], y[:1000])

#    rnn.gradient_check([0,1,2,3], [1,2,3,4])



