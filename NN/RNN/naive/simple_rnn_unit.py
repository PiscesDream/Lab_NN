import theano 
import theano.tensor as T
import numpy as np
from sklearn.cross_validation import train_test_split

def generate_wb(dim1, dim2, label): # generate a set of w and b for transformation from dim1 to dim2
    upper = 6/np.sqrt(dim1+dim2)
    lower = -upper 
    w = theano.shared(name='w_{}'.format(label),
                      value=np.random.uniform(upper, lower, size=(dim1, dim2)).astype(theano.config.floatX))
    b = theano.shared(name='b_{}'.format(label),
                      value=np.random.uniform(upper, lower, size=(dim2)).astype(theano.config.floatX))
    return w, b

class RNN(object):
    def __init__(self, dim_input, dim_hidden, dim_output, activation=T.tanh, bptt_truncate=4):

        x = T.tensor3('x') # number of sequence * time span * feature size
        y = T.ivector('y')  # only one label (last one) is available
                           # (mask = T.matrix('mask')  # 1 1 1 1 0 0)
        lr = T.scalar('lr', dtype=theano.config.floatX)

        w_input, b_input   = generate_wb(dim_input, dim_hidden, 'input')
        w_hidden, b_hidden = generate_wb(dim_hidden, dim_hidden, 'hidden')
        w_output, b_output = generate_wb(dim_hidden, dim_output, 'output')

        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        def forward(x_t, s_prev, w_i, b_i, w_h, b_h, w_o, b_o): 
            # current input, previous hidden state, w_input, w_hidden, w_output
            # x.shape = n * feature size (scaning on time span)

#           i = x_t.dot(w_i) + b_i                          # n * dim_i
            i = activation( x_t.dot(w_i) + b_i )            # n * dim_i

            s = activation( i + s_prev.dot(w_h) + b_h )     # n * dim_h
#           o = T.nnet.softmax( s.dot(w_o) + b_o )              # n * dim_o 
#           return s, o # n*dim_h, n*dim_o
            return s # n*dim_h,  o is on longer needed because only the last label is given

        # T*n*dim_h
        s,updates = theano.scan(
            fn=forward,
            sequences = x.swapaxes(0, 1),
            outputs_info = T.zeros((x.shape[0], dim_hidden)),
            non_sequences = [w_input, b_input, w_hidden, b_hidden, w_output, b_output],
            truncate_gradient=bptt_truncate,
            strict = True)

        output = T.nnet.softmax(activation(s[-1].dot(w_output) + b_output))
        prediction = output.argmax(1)
        loss = T.sum(T.nnet.categorical_crossentropy(output, y))

        params = [w_input, b_input, w_hidden, b_hidden, w_output, b_output]
        updates = map(lambda x: (x, x-lr*T.grad(loss, x)), params)

        print 'compile step()'
        self.step = theano.function([x, y, lr], [loss], updates=updates)
#       print 'compile error()'
#       self.error = theano.function([x, y], loss)
#       print 'compile forward()'
#       self.forward = theano.function([x], [s.swapaxes(0,1), output])
        print 'compile predict()'
        self.predict = theano.function([x], prediction)

    def fit(self, x, y, lr=1e-2,
            max_iter=100000, 
            test_iter=100,      # test on validation set
            disp_iter=1,        # display
            lr_iter=10,         # update lr
            val=None):
        valx, valy = val if val !=None else (x, y)

        lastcost = np.inf
        cost = []
        for i in xrange(max_iter):
            if i%test_iter == 0:
                correct = (self.predict(valx) == valy).sum()
                print '\n\tacc = {}/{} = {}'.format(correct, len(valy), float(correct)/len(valy))
            if i % lr_iter == 0:
                cost = np.mean(cost)
                lr = lr * 0.8 if cost >= lastcost else lr * 1.02
                lastcost = cost
                cost = []

            # update
            cost.append( self.step(x, y, lr) )

            if i % disp_iter == 0: 
                print 'Iter[{}] lr={}'.format(i, lr)
                print '\tcost: {}'.format(np.mean(cost))
 





def demo():
    x = np.random.uniform(0, 1, size=(100, 5, 3)).astype(theano.config.floatX) # n * T * d_i
    y = np.random.randint(10, size=(100)).astype('int32')

    net = RNN(3, 7, 10) # d_i, d_h, d_o

    loss = net.error(x, y)
    s, output = net.forward(x)
    pred = net.predict(x)
    print s.shape           # n * T * d_h 
    print output.shape      # n * d_o
    print pred.tolist()
    assert -np.log(output[np.arange(len(y)), y]).sum() == loss 

if __name__ == '__main__':
    d_i = 10
    Time = 10
    n_per_class = 500
    classes = 10

    import common
    x, y = common.generate_random_walk(classes, n_per_class, d_i, Time, decay=1.00)
    print x.shape
    print y.shape
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.33)

    net = RNN(d_i, 4000, classes, bptt_truncate=-1)
    net.fit(trainx, trainy, lr=1e-4, 
            max_iter=100000, 
            test_iter=100,       # test on validation set
            disp_iter=10,        # display
            lr_iter=20,        # update lr
            val = (testx, testy)
            )




    


