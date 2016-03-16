from preprocess import load_data
#import pyximport; pyximport.install()
#from npmodel import npRNN
from Tmodel import TheanoRNN as RNN
from Tmodel import TheanoGRU as GRU
from config import *
import sys



def translate(sentence, index_to_word):
    return ' '.join(map(lambda x: index_to_word[x], sentence))

if __name__ == '__main__':
    global index_to_word, word_to_index  

    word_dim = 3000
    hidden_dim = 100
    sc = 10000

    x, y, index_to_word = load_data(vocabulary_size = word_dim, sentences_count = sc)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    sys.stdout.flush()
    
    rnn = GRU(word_dim, dictionary=index_to_word)
    rnn.load('./models/rnn-theano-100-3000-2016-01-16-09-31-26.npz')

    while True:
        current_sent = [word_to_index[sentence_start_token]]
        print 'reset sentence'
        while current_sent[-1] != word_to_index[sentence_end_token]:
            print '='*100
            print '\tCurrent sentence: {}'.format(translate(current_sent, index_to_word))

            prob = rnn.forward(current_sent)[-1]
            ind = prob.argsort()[-1:-10:-1]
            print '\tPredict words: {}'.format(zip(prob[ind], map(lambda x: index_to_word[x], ind)) )

            word = None
            while word == None:
                word = raw_input("\tInput next word:")
                word = word_to_index.get(word.strip(), None)

            current_sent.append(word)
