from NN.CNN import ConvolutionLayer
from NN.RNN.naive import RNN

from NN.common.layers import InputLayer, FullConnectLayer, SoftmaxLayer
from NN.common.toolkits import generate_wb 
import sys

import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams

class AttentionUnit(object):
    # RNN + location
    def __init__(self, x, glimpse_shape, glimpse_times, dim_hidden, rng, rng_std=1.0, activation=T.tanh, bptt_truncate=4, name='AttentionModel'):
        # random for rng
        self.rng = rng
        self.rng_std = rng_std
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
#           loc_mean = activation( s_prev.dot(w_l) + b_l )  # n * 2
            loc_mean = s_prev.dot(w_l) + b_l  # n * 2  TODO
            # glimpse
            glimpse, loc = self._glimpse(x, loc_mean) # n * dim_hidden, n * 2
            # input
            s = activation( glimpse.dot(w_i) + s_prev.dot(w_h) + b_h ) # n * dim_hidden
            return s, loc, loc_mean # n*dim_h, n * 2, n * 2

        [s, loc, loc_mean], updates = theano.scan(
            fn=forward,
            sequences = T.arange(glimpse_times), #x.swapaxes(0, 1),
            outputs_info = [T.zeros((x.shape[0], dim_hidden)), None, None], 
            non_sequences = [x, w_input, w_hidden, b_hidden, w_location, b_location],
            truncate_gradient=bptt_truncate,
            strict = True)
        # s: Time * n * dim_hidden
        # loc: Time * n * 2

        self.output = s.swapaxes(0, 1) # N * Time * dim_hidden
        self.location = loc.swapaxes(0, 1) # N * T * dim_h
        self.location_mean = loc_mean.swapaxes(0, 1) # N * T * 2
        self.location_logp = (-T.log(T.sqrt(2*np.pi)*rng_std)-((loc-loc_mean)**2)/(2.0*rng_std**2) ).swapaxes(0,1)
#       self.location_logp = - float(1.0/(2.0*rng_std**2)) * ((loc-loc_mean)**2).swapaxes(0,1)
                # this part is useless in training >> - T.log(T.sqrt(2*T.pi)*rng_std) 
        self.params = [w_input, w_hidden, b_hidden]
        self.reinforceParams = [w_location, b_location]

    def _glimpse(self, x, loc_mean):
        '''
            x: tensor3 (N, W, H)
            raw_loc: matrix (N, 2) mean 
            loc: matrix (N, 2) center point
        '''
#       loc = T.cast(T.maximum(loc_mean*x.shape[1:], 0), 'int32')
#       loc = T.cast(self.rng.normal(size=loc_mean.shape, avg=loc_mean, std=self.rng_std), 'int32')

        loc = self.rng.normal(size=(x.shape[0], 2), avg=loc_mean, std=self.rng_std)
        loc = T.cast(T.maximum(T.round(loc), 0), 'int32') 
        
        locx = T.minimum(loc[:,0], x.shape[1]-self.glimpse_shape[0]-1)  #*self.glimpse_shape[2]
        locy = T.minimum(loc[:,1], x.shape[2]-self.glimpse_shape[1]-1)  #*self.glimpse_shape[2] 
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
        loc = T.stack(locx+self.glimpse_shape[0]*self.glimpse_shape[2]/2, 
                      locy+self.glimpse_shape[1]*self.glimpse_shape[2]/2).T
        # x: N * dim_in
        # loc: N * 2
        return x, loc # crop


class AttentionModel(object):
    def __init__(self, glimpse_shape, glimpse_times, dim_hidden, dim_fc, dim_out, rng_std=1.0, activation=T.tanh, bptt_truncate=-1, reward_coef=1.00):
        x = T.ftensor3('x')  # N * W * H 
        y = T.ivector('y')  # label 
        lr = T.fscalar('lr')
        rng = MRG_RandomStreams(np.random.randint(9999999))
#       rng = theano.tensor.shared_randomstreams.RandomStreams(np.random.randint(9999999))

        reward_coef = T.cast(theano.shared(value=reward_coef, name='reward_coef'), theano.config.floatX)
    
        i = InputLayer(x)
        au = AttentionUnit(x, glimpse_shape, glimpse_times, dim_hidden, rng, rng_std, activation, bptt_truncate)
#       layers = [i, au, InputLayer(au.output[:,:,:].flatten(2))]
#       dim_fc = [glimpse_times*dim_hidden] + dim_fc + [dim_out]
        layers = [i, au, InputLayer(au.output[:,-1,:].flatten(2))]
        dim_fc = [dim_hidden] + dim_fc + [dim_out]
        for Idim, Odim in zip(dim_fc[:-1], dim_fc[1:]):
            fc = FullConnectLayer(layers[-1].output, Idim, Odim, activation, 'FC')
            layers.append(fc)
        sm = SoftmaxLayer(layers[-1].output)
        layers.append(sm)

        output = sm.output       # N * classes 
        hidoutput = au.output    # N * dim_output 
        location = au.location   # N * T * dim_hidden
        prediction = output.argmax(1) # N

        # calc
        equalvec = T.eq(prediction, y)
        correct = T.cast(T.sum(equalvec), 'float32')
        noequalvec = T.neq(prediction, y)
        nocorrect = T.cast(T.sum(noequalvec), 'float32')
        LogLoss = T.log(output)[T.arange(y.shape[0]), y][equalvec]
        
        # gradient descent
        gdobjective = LogLoss.sum()/x.shape[0]  # correct * dim_output (only has value on the correctly predicted sample)
        gdparams = reduce(lambda x, y: x+y.params, layers, []) 
        gdupdates = map(lambda x: (x, x+lr*T.grad(gdobjective, x)), gdparams)

        # reinforce learning
#       rlobjective = au.location_logp[equalvec].sum()/x.shape[0]
        # LogLoss is reward, gradient is gradient
        rlobjective = reward_coef*au.location_logp[equalvec][:,:,:].sum() /x.shape[0]
        rlparams = au.reinforceParams 
        rlupdates = map(lambda x: (x, x+lr*T.grad(rlobjective, x)), rlparams)
         
        print 'compile step()'
        self.step = theano.function([x, y, lr], [gdobjective, rlobjective, correct], updates=gdupdates+rlupdates)
    #       print 'compile gdstep()'
    #       self.gdstep = theano.function([x, y, lr], [gdobjective, correct, location], updates=gdupdates)
    #       print 'compile rlstep()'
    #       self.rlstep = theano.function([x, y, lr], [rlobjective], updates=rlupdates)
        print 'compile predict()'
        self.predict = theano.function([x], prediction)
        print 'compile forward()'
        self.forward = theano.function([x], map(lambda x: x.output, layers)) #[layers[-3].output, fc.output])
        print 'compile error()'
        self.error = theano.function([x, y], gdobjective)
        print 'compile locate()'
        self.locate = theano.function([x], location) #[layers[-3].output, fc.output])
        print 'compile debug()'
        self.debug = theano.function([x, y, lr], au.location_mean, on_unused_input='warn')


    def fit(self, x, y, 
            batch_size,
            lr=1e-2,
            max_iter=100000, 
            test_iter=1000,      # test on validation set
            disp_iter=10,        # display
            lr_iter=100,         # update lr
            decay=0.9,
            val=None):
        valx, valy = val if val !=None else (x, y)

        batch_count = len(x)/batch_size
        
        np.set_printoptions(precision=3)
        def npinline(x):
            return map(lambda loc: tuple(map(lambda loci: round(loci, 3), loc)), x.tolist())

        lastcost = np.inf
        gdcost = []
        rlcost = []
        correct = []
        for i in xrange(max_iter):
            if i%test_iter == 0:
                cor = (self.predict(valx) == valy).sum()
                print '\n\tacc = {}/{} = {}'.format(cor, len(valy), float(cor)/len(valy))
               #print zip(valy, self.predict(valx))
            if i % lr_iter == 0:
                lr *= decay
                pass

            # update
            batch_index = np.random.randint(batch_count)
            start = batch_index*batch_size
            end = (batch_index+1)*batch_size

            gdcosti, rlcosti, correcti = self.step(x[start:end], y[start:end], lr)
            gdcost.append(gdcosti)
            rlcost.append(rlcosti)
            correct.append(correcti)

            if i % disp_iter == 0: 
                print 'Iter[{}] lr={}'.format(i, lr)
                print '\tGDcost: {}\tRLcost: {}\tcorrect: {}/{}'.\
                    format(np.mean(gdcost), np.mean(rlcost), np.mean(correct), batch_size) 
                print '\ty: {}'.format(y[0])
                print '\tloc_mean: {}'.format( npinline(self.debug([x[0]], [y[0]], lr).reshape(-1, 2)) )
                print '\tlocation: {}'.format( npinline(self.locate([x[0]]).reshape(-1, 2)) )
#               print '\tloc_mean: {}'.format( npinline(self.debug(x[start:end], y[start:end], lr)[0].reshape(-1, 2)) )
#               print '\tlocation: {}'.format( npinline(self.locate(x[start:end])[0].reshape(-1, 2)) )
                sys.stdout.flush()
                cost = []
                correct = [] 
 

