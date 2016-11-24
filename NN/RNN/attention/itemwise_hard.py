# item-wise hard attention
from NN.common.layers import InputLayer, FullConnectLayer, SoftmaxLayer
from NN.common.toolkits import generate_wb 
from NN.common.nets import NetModel
import sys

import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams

class AttentionUnit(object):
    # LSTM + location
    # x: (N, item_count, dim_input)
    def __init__(self, x, item_count, dim_input, glimpse_times, dim_hidden, rng, activation=T.tanh, bptt_truncate=-1, name='AttentionModel', minimum_p=1e-10):
        '''
            Itemwise hard attention

            Only one item with dim_input will be considered each glimpse.
        '''
        # random for rng
        self.rng = rng
        self.glimpse_times = glimpse_times

#       W_x0 = generate_wb(dim_input, 4*dim_hidden, '{}_x'.format(name), params=['w'])
#       W_h0, b_h0 = generate_wb(dim_hidden, 4*dim_hidden, '{}_hidden'.format(name))
        W_x = generate_wb(dim_input, 4*dim_hidden, '{}_x'.format(name), params=['w'])
        W_c = generate_wb(dim_hidden, 3*dim_hidden, '{}_c'.format(name), params=['w'])
        W_h, b_h = generate_wb(dim_hidden, 4*dim_hidden, '{}_hidden'.format(name))

        w_location, b_location = generate_wb(dim_hidden, item_count, '{}->item'.format(name), params=['w', 'b'])

        # example
#       In [121]: y, u = theano.scan(fn=lambda p, x: rng.choice(size=(1,), a=x, p=p)[0], sequences=p/p.sum(1).dimshuffle(0, 'x'), outputs_info=None, non_sequences=x)
#       In [122]: f = theano.function([p, x], y, updates=u)

#       def get_pick(p, item_count):
#           return rng.choice(size=[1], a=item_count, p=p)[0]

        def forward(times, s_prev, C_prev, x, W_x, W_c, W_h, b_h, w_l, b_l): #, w_l0, b_l0): 
            # current input, previous hidden state, w_input, w_hidden, w_output

            # get item p 
            item_p = T.nnet.softmax( s_prev.dot(w_l) + b_l )  # n * item_count 
            # max pick
            # picked = item_p.argmax(1)

            # random pick, something wrong here
#           picked, _ = theano.scan(fn=get_pick, 
#                               sequences=[item_p],
#                               outputs_info=None,
#                               non_sequences=[x.shape[2]],
#                               strict=True)

            # random pick via threshold
            accp, _ =theano.scan(fn=lambda last, current: last+current, 
                    sequences=item_p.T, 
                    outputs_info=T.zeros((item_p.shape[0],)), 
                    strict=True) # item_count * N
            thres = rng.uniform(size=(x.shape[0],1), low=0, high=1, dtype=theano.config.floatX)
            picked = (-1.0/(accp.T-thres)).argmin(1)

            items = x[T.arange(x.shape[0]), times, picked] # n * dim_input

            # LSTM
            res    =   items.dot(W_x) +  s_prev.dot(W_h)      + b_h.dimshuffle('x', 0) 
            peephole = C_prev.dot(W_c)
            f = T.nnet.sigmoid(res[:, 0*dim_hidden:1*dim_hidden] + peephole[:, 0*dim_hidden:1*dim_hidden]) # N * dh
            i = T.nnet.sigmoid(res[:, 1*dim_hidden:2*dim_hidden] + peephole[:, 1*dim_hidden:2*dim_hidden]) # N * dh
            C_hat =     T.tanh(res[:, 2*dim_hidden:3*dim_hidden]) # N * dh
            o = T.nnet.sigmoid(res[:, 3*dim_hidden:4*dim_hidden] + peephole[:, 2*dim_hidden:3*dim_hidden]) # N * dh
            C = f*C_prev + i*C_hat # N * dh
            s = o * T.tanh(C)      # N * dh
            return s, C, items, picked, item_p

        [s, C, items, picked, item_p], updates = theano.scan(
            fn=forward,
#           sequences = x.swapaxes(0, 1),  #x:  Time * N * item count * item feature len(1)
            sequences = T.arange(glimpse_times), #x.swapaxes(0, 1),
            outputs_info = [T.zeros((x.shape[0], dim_hidden)), 
                            T.zeros((x.shape[0], dim_hidden)), 
                            None, None, None], 
            non_sequences = [x, W_x, W_c, W_h, b_h, w_location, b_location],#w_location0, b_location0],
            truncate_gradient=bptt_truncate,
            strict = True)

        self.output = s.swapaxes(0, 1) # N * Time * dim_hidden
        self.cell = C.swapaxes(0, 1)

        self.params = [W_x, W_c, W_h, b_h]
        self.reinforceParams = [w_location, b_location] #, w_location0, b_location0]

        # for debug
        self.item_p = item_p.swapaxes(0, 1)  # N * Time * item_count 
        self.picked = picked.swapaxes(0, 1) # N * Time * item_count
        self.items = items.swapaxes(0, 1) # N * Time * dim_input 

class AttentionModel(NetModel):
    def __init__(self, 
        item_count, dim_input, 
        glimpse_times, 
        dim_hidden, dim_fc, dim_out, 
        reward_base, 
        activation=T.tanh, bptt_truncate=-1, 
        lmbd=0.1, # gdupdate + lmbd*rlupdate
        DEBUG=False,
        ): 
#       super(AttentionUnit, self).__init__()

        if reward_base == None: 
            reward_base = np.zeros((glimpse_times)).astype('float32')
            reward_base[-1] = 1.0

        x = T.ftensor4('x') # x: N * Time spac: * item count * item feature len(1)  # old x: (N, item_count, dim_input)
        y = T.ivector('y')  # label 
        lr = T.fscalar('lr')
        reward_base = theano.shared(name='reward_base', value=np.array(reward_base).astype(theano.config.floatX), borrow=True) # Time (vector)
        reward_bias = T.fvector('reward_bias')
        rng = T.shared_randomstreams.RandomStreams(123)
#       rng = MRG_RandomStreams(np.random.randint(9999999))
        self.glimpse_times = glimpse_times
    
        i = InputLayer(x)
        au = AttentionUnit(x, item_count, dim_input, glimpse_times, dim_hidden, rng, activation, bptt_truncate)
        layers = [i, au]
#   only last state counts
        layers.append(InputLayer(au.output[:,-1,:]) ) 
        dim_fc = [dim_hidden] + dim_fc + [dim_out]
#   all states count
#       layers.append(InputLayer(au.output[:,:,:].flatten(2)) ) 
#       dim_fc = [dim_hidden*glimpse_times] + dim_fc + [dim_out]

        for Idim, Odim in zip(dim_fc[:-1], dim_fc[1:]):
            fc = FullConnectLayer(layers[-1].output, Idim, Odim, activation, 'FC')
            layers.append(fc)
        sm = SoftmaxLayer(layers[-1].output)
        layers.append(sm)

        output = sm.output       # N * classes 
        hidoutput = au.output    # N * dim_output 
        prediction = output.argmax(1) # N

        # calc
        equalvec = T.eq(prediction, y) # [0, 1, 0, 0, 1 ...]
        correct = T.cast(T.sum(equalvec), 'float32')
        logLoss = T.log(output)[T.arange(y.shape[0]), y] 
        reward_biased = T.outer(equalvec, reward_base) - reward_bias.dimshuffle('x', 0) # N * Time
        
        # gradient descent
        gdobjective = logLoss.sum()/x.shape[0]  # correct * dim_output (only has value on the correctly predicted sample)
        gdparams = reduce(lambda x, y: x+y.params, layers, []) 
        gdupdates = map(lambda x: (x, x+lr*T.grad(gdobjective, x)), gdparams)

        # reinforce learning
        # without maximum, then -log(p) will decrease the total p
#       rlobjective = (reward_biased.dimshuffle(0, 1, 'x') * T.log(au.item_p)).sum() / x.shape[0]
        rlobjective = (T.maximum(reward_biased.dimshuffle(0, 1, 'x'), 0) * T.log(au.item_p)).sum() / correct 
            # item_p: N * Time * item_count 
            # reward_biased: N * Time
        rlparams = au.reinforceParams 
        rlupdates = map(lambda x: (x, x+lr*lmbd*T.grad(rlobjective, x)), rlparams)

        # Hidden state keeps unchange in time
        deltas = T.stack(*[((au.output[:,i,:].mean(0)-au.output[:,i+1,:].mean(0))**2).sum()  for i in xrange(glimpse_times-1)])
            # N * Time * dim_hidden
         
        print 'compile step()'
        self.step = theano.function([x, y, lr, reward_bias], [gdobjective, rlobjective, correct, T.outer(equalvec, reward_base)], updates=gdupdates+rlupdates+rng.updates())
    #       print 'compile gdstep()'
    #       self.gdstep = theano.function([x, y, lr], [gdobjective, correct, location], updates=gdupdates)
    #       print 'compile rlstep()'
    #       self.rlstep = theano.function([x, y, lr], [rlobjective], updates=rlupdates)
        print 'compile predict()'
        self.predict = theano.function([x], prediction)
        print 'compile picked()'
        self.picked = theano.function([x], au.picked) # item indices 
        print 'compile item_p()'
        self.item_p = theano.function([x], au.item_p) # item indices 
        if DEBUG:
            print 'compile error()'
            self.error = theano.function([x, y, reward_bias], [gdobjective, rlobjective])
            print 'compile forward()'
            self.forward = theano.function([x], map(lambda x: x.output, layers)) #[layers[-3].output, fc.output])
#           print 'compile glimpse()'
#           self.glimpse = theano.function([x], au.glimpse) #[layers[-3].output, fc.output])
#           print 'compile innerstate()'
#           self.getinnerstate = theano.function([x], au.innerstate)
#           print 'compile locate()'
#           self.locate = theano.function([x], [au.location_mean, location]) #[layers[-3].output, fc.output])
#           print 'compile debug()'
#           self.debug = theano.function([x, y, lr, reward_bias], [deltas, au.location_p], on_unused_input='warn')

        # self.xxx
        self.layers = layers
        self.params = gdparams + rlparams

    def fit(self, x, y, 
            batch_size,
            lr=1e-2,
            max_iter=100000, 
            test_iter=1000,      # test on validation set
            disp_iter=10,        # display
            lr_iter=100,         # update lr
            reward_iter=100,
            reward_base=None,
            decay=0.9,
            gamma=0.80, # the reward bias update speed
            quick_save=(0, 'temp.model'),
            val=None):
        valx, valy = val if val !=None else (x, y)

        batch_count = len(x)/batch_size

        lastcost = np.inf
        gdcost = []
        rlcost = []
        correct = []
        reward = np.zeros((self.glimpse_times)).astype('float32')
        for i in xrange(max_iter):
            if i%test_iter == 0:
                cor = (self.predict(valx) == valy).sum()
                print '\n\tacc = {}/{} = {}'.format(cor, len(valy), float(cor)/len(valy))
               #print zip(valy, self.predict(valx))
            if i % lr_iter == 0:
                lr *= decay
                pass
#           if i % reward_iter == 0:
#               reward = []
            if quick_save[0] != 0 and i % quick_save[0] == 0:
                print 'saving ...'
                self.save_params(quick_save[1], verbose=1)

            # update
            batch_index = np.random.randint(batch_count)
            start = batch_index*batch_size
            end = (batch_index+1)*batch_size

            inputs = (x[start:end], y[start:end], lr, reward)
            gdcosti, rlcosti, correcti, rewardi = self.step(*inputs)
            gdcost.append(gdcosti)
            rlcost.append(rlcosti)
            correct.append(correcti)
            reward = gamma*reward + (1-gamma)*rewardi.mean(0) # N * Time -> Time

            if i % disp_iter == 0: 
                print 'Iter[{}] lr={}'.format(i, lr)
                print '\tGDcost: {}\tRLcost: {}\tcorrect: {}/{}\treward_bias: {}'.\
                    format(np.mean(gdcost), np.mean(rlcost), np.mean(correct), batch_size, reward) 
                picked = self.picked(x[start:end])[:10]
                item_p = self.item_p(x[start:end])[:10]
                for i in xrange(10):
                    print picked[i], item_p[i, np.arange(self.glimpse_times), picked[i]]

                sys.stdout.flush()
                cost = []
                correct = [] 
 

