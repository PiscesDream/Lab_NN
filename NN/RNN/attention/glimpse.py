from NN.CNN import ConvolutionLayer
from NN.RNN.naive import RNN

from NN.common.layers import InputLayer, FullConnectLayer, SoftmaxLayer
from NN.common.toolkits import generate_wb 

import numpy as np
import theano
import theano.tensor as T

class AttentionUnit(object):
    # RNN + location
    def __init__(self, x, glimpse_shape, glimpse_times, dim_hidden, activation=T.tanh, bptt_truncate=4, name='AttentionModel'):
        # n * W * H --> n * dim_input --> n * dim_hidden
        self.glimpse_shape = glimpse_shape
        dim_input = np.prod(glimpse_shape)
        w_input = generate_wb(dim_input, dim_hidden, '{}->input'.format(name), params=['w'])
        w_hidden, b_hidden = generate_wb(dim_hidden, dim_hidden, '{}->hidden'.format(name), params=['w', 'b'])
        w_location, b_location = generate_wb(dim_hidden, 2, '{}->location'.format(name), params=['w', 'b'])

        def forward(times, s_prev, x, w_i, w_h, b_h, w_l, b_l): 
            # current input, previous hidden state, w_input, w_hidden, w_output
            # x.shape = n * W * H 

            # get location vector
            loc = activation( s_prev.dot(w_l) + b_l )  # n * 2
            # glimpse
            glimpse, loc = self._glimpse(x, loc) # n * dim_hidden, n * 2
            # input
            s = activation( glimpse.dot(w_i) + s_prev.dot(w_h) + b_h ) # n * dim_hidden
            return s, loc # n*dim_h, n * 2

        [s, loc] ,updates = theano.scan(
            fn=forward,
            sequences = T.arange(glimpse_times), #x.swapaxes(0, 1),
            outputs_info = [T.zeros((x.shape[0], dim_hidden)), None], 
            non_sequences = [x, w_input, w_hidden, b_hidden, w_location, b_location],
            truncate_gradient=bptt_truncate,
            strict = True)
        # s: Time * n * dim_hidden
        # loc: Time * n * 2

        self.output = s.swapaxes(0, 1) # N * Time * dim_hidden
        self.location = loc.swapaxes(0, 1) # N * T * dim_h
        self.params = [w_input, w_hidden, b_hidden]
        self.reinforceParams = [w_location, b_location]

    def _glimpse(self, x, loc):
        '''
            x: tensor3 (N, W, H)
            loc: matrix (N, 2) (inclusive upper-left corner)
        '''
        loc = T.cast(T.maximum(loc*x.shape[1:], 0), 'int32')
        
        locx = T.minimum(loc[:,0], x.shape[1]-self.glimpse_shape[0]*self.glimpse_shape[2])
        locy = T.minimum(loc[:,1], x.shape[2]-self.glimpse_shape[1]*self.glimpse_shape[2])
        def glimpse_each(xi, locxi, locyi):
            return xi[locxi:locxi+self.glimpse_shape[0], locyi:locyi+self.glimpse_shape[1]].flatten()
#           g = []
#           upper = locxi
#           bottom = locxi+self.glimpse_shape[2]*self.glimpse_shape[0]
#           left = locyi
#           right = locyi+self.glimpse_shape[2]*self.glimpse_shape[1]
#           for level in range(1, self.glimpse_shape[2]+1)[::-1]:
#               gi = xi[upper:bottom, left:right]
#               g.append( T.signal.downsample.max_pool_2d(gi, (level, level)) )

#               upper += self.glimpse_shape[0]/2
#               bottom -= self.glimpse_shape[0]/2
#               left += self.glimpse_shape[1]/2
#               right -= self.glimpse_shape[1]/2
#           return T.stack(g).flatten()

        x, updates = theano.scan(
            fn = glimpse_each,
            sequences = [x, locx, locy],
            strict=True)
        loc = T.stack(locx+self.glimpse_shape[2]*self.glimpse_shape[0]/2, 
                      locy+self.glimpse_shape[1]*self.glimpse_shape[0]/2).T
        # x: N * dim_in
        # loc: N * 2
        return x, loc # crop


class AttentionModel(object):
    def __init__(self, glimpse_shape, glimpse_times, dim_hidden, dim_fc, dim_out, activation=T.tanh, bptt_truncate=-1, reward_coef=1.00):
        x = T.ftensor3('x')  # N * W * H 
        y = T.ivector('y')  # label 
        lr = T.fscalar('lr')
        reward_coef = T.cast(theano.shared(value=reward_coef, name='reward_coef'), theano.config.floatX)
    
        i = InputLayer(x)
        au = AttentionUnit(x, glimpse_shape, glimpse_times, dim_hidden, activation, bptt_truncate)
        layers = [i, au, InputLayer(au.output[:,:,:].flatten(2))]
        dim_fc = [glimpse_times*dim_hidden] + dim_fc + [dim_out]
        for Idim, Odim in zip(dim_fc[:-1], dim_fc[1:]):
            fc = FullConnectLayer(layers[-1].output, Idim, Odim, activation, 'FC')
            layers.append(fc)
        sm = SoftmaxLayer(layers[-1].output)
        layers.append(sm)

        output = sm.output       # N * dim_output
        location = au.location   # N * T * dim_hidden
        prediction = output.argmax(1) # N
        correct = T.sum(T.eq(prediction, y))
        
        # gradient descent
        reward = reward_coef * T.sum( -T.log(output)[T.eq(prediction, y)] ) # correct * dim_output (only has value on the correctly predicted sample)
        reward = reward/correct
        gdparams = reduce(lambda x, y: x+y.params, layers, []) 
        gdupdates = map(lambda x: (x, x-lr*T.grad(reward, x)), gdparams)

        # reinforce learning
        rlparams = au.reinforceParams 
        rlupdates = map(lambda x: (x, x - T.cast(lr*correct*reward_coef*x, 'float32')), rlparams)
#       rlupdates = []
        
        updates = gdupdates+rlupdates
        print 'compile step()'
        self.step = theano.function([x, y, lr], [reward, correct], updates=updates)
        print 'compile error()'
        self.error = theano.function([x, y], reward)
        print 'compile forward()'
        self.forward = theano.function([x], map(lambda x: x.output, layers)) #[layers[-3].output, fc.output])
        print 'compile locate()'
        self.locate = theano.function([x], location) #[layers[-3].output, fc.output])
        print 'compile predict()'
        self.predict = theano.function([x], prediction)


    def fit(self, x, y, 
            batch_size,
            lr=1e-2,
            max_iter=100000, 
            test_iter=1000,      # test on validation set
            disp_iter=10,        # display
            lr_iter=100,         # update lr
            val=None):
        valx, valy = val if val !=None else (x, y)

        batch_count = len(x)/batch_size

        lastcost = np.inf
        cost = []
        correct = []
        for i in xrange(max_iter):
            if i%test_iter == 0:
                cor = (self.predict(valx) == valy).sum()
                print '\n\tacc = {}/{} = {}'.format(cor, len(valy), float(cor)/len(valy))
               #print zip(valy, self.predict(valx))
            if i % lr_iter == 0:
                lr *= 0.8
                pass

            # update
            batch_index = np.random.randint(batch_count)
            start = batch_index*batch_size
            end = (batch_index+1)*batch_size
            costi, correcti = self.step(x[start:end], y[start:end], lr)
            cost.append(costi)
            correct.append(correcti)

            if i % disp_iter == 0: 
                print 'Iter[{}] lr={}'.format(i, lr)
                print '\treward: {} \t correct: {}/{}'.format(np.mean(cost), np.mean(correct), batch_size)
                cost = []
                correct = [] 
 
