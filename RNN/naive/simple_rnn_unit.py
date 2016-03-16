import theano 
import theano.tensor as T

class RNN(object):
    def __init__(self, hidden_len=100):

        x = T.matrix('x')  # x1 x2 x3 ...  fixed length
        y = T.vector('y')  # only one label (last one) is available
#       mask = T.matrix('mask')  # 1 1 1 1 0 0 

        w_hidden = T.matrix('w_hidden')
        w_output = T.matrix('w_output')
        w_input = T.matrix('w_input')

if __name__ == '__main__':
    pass
    
