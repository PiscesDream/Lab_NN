import numpy as np
import operator, sys
from datetime import datetime
from config import *

import theano as theano
import theano.tensor as T

def randinit(dim1, dim2, dim3=None):
    if dim3:
        shape = (dim3, dim1, dim2)
    else:
        shape = (dim1, dim2)
    return np.random.uniform(-np.sqrt(1./dim2), np.sqrt(1./dim2), shape).\
        astype(theano.config.floatX)

class TheanoRNN:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4, dictionary=None):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.dictionary = dictionary

        self.U = theano.shared(randinit(hidden_dim, word_dim)  , 'U')
        self.V = theano.shared(randinit(word_dim, hidden_dim)  , 'V')
        self.W = theano.shared(randinit(hidden_dim, hidden_dim), 'W')      

        self.__build__()

    def __build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')
        y = T.ivector('y')
        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        def forward(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:, x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
        [o,s], updates = theano.scan(
            forward,
            sequences = x, # the sentences
            outputs_info = [None, dict(initial=T.zeros(self.hidden_dim))], # default result
            non_sequences = [U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
 
        # Gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.U, self.U - learning_rate * dU),
                              (self.V, self.V - learning_rate * dV),
                              (self.W, self.W - learning_rate * dW)])
        self.forward = theano.function([x], o)
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)   


    def gradient_check(model, x, y, h=0.001, error_threshold=0.01):
        # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
        model.bptt_truncate = 1000
        # Calculate the gradients using backprop
        bptt_gradients = model.bptt(x, y)
        # List of all parameters we want to chec.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter_T = operator.attrgetter(pname)(model)
            parameter = parameter_T.get_value()
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                parameter_T.set_value(parameter)
                gradplus = model.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                parameter_T.set_value(parameter)
                gradminus = model.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                parameter[ix] = original_value
                parameter_T.set_value(parameter)
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return 
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)

    def fit(self, X_train, y_train, learning_rate=0.005, nepoch=1, nstep=10000, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                
                print "="*100
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                for i in xrange(100):
                    try:
                        print "Generate[{}]: ".format(i), ' '.join(self.generate(self.dictionary))
                    except:
                        pass

                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5  
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
                # Saving model oarameters
                self.save("./models/rnn-theano-%d-%d-%s.npz" % (self.hidden_dim, self.word_dim, time))
            # For each training example...

            for i in np.random.randint(0, len(y), size=nstep):
                # One SGD step
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1


    def save(model, outfile):
        U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
        np.savez(outfile, U=U, V=V, W=W)
        print "Saved model parameters to %s." % outfile
       
    def load(model, path):
        npzfile = np.load(path)
        U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
        model.hidden_dim = U.shape[0]
        model.word_dim = U.shape[1]
        model.U.set_value(U)
        model.V.set_value(V)
        model.W.set_value(W)
        print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])

    def generate(model, index_to_word, maxlen=20):
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        # We start the sentence with the start token
        new_sentence = [word_to_index[sentence_start_token]]
        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[sentence_end_token]\
                and len(new_sentence) <= maxlen:
            next_word_probs = model.forward(new_sentence)
            sampled_word = word_to_index[unknown_token]
            # We don't want to sample unknown words
            while sampled_word == word_to_index[unknown_token]:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [index_to_word[x] for x in new_sentence]
        return sentence_str























class TheanoGRU:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=-1, dictionary=None):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.dictionary = dictionary

        # Initialize the network parameters
        E = randinit(hidden_dim, word_dim)
        # input?
        U = randinit(hidden_dim, hidden_dim, 6)
        # circle
        W = randinit(hidden_dim, hidden_dim, 6) 
        # output
        V = randinit(word_dim, hidden_dim) 
        # bias
        b = np.zeros((6, hidden_dim))
        # ?
        c = np.zeros(word_dim)
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E)
        self.U = theano.shared(name='U', value=U)
        self.W = theano.shared(name='W', value=W)
        self.V = theano.shared(name='V', value=V)
        self.b = theano.shared(name='b', value=b)
        self.c = theano.shared(name='c', value=c)
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros_like(E))
        self.mU = theano.shared(name='mU', value=np.zeros_like(U))
        self.mV = theano.shared(name='mV', value=np.zeros_like(V))
        self.mW = theano.shared(name='mW', value=np.zeros_like(W))
        self.mb = theano.shared(name='mb', value=np.zeros_like(b))
        self.mc = theano.shared(name='mc', value=np.zeros_like(c))

        self.params = [(x.name, x) for x in [self.E, self.U, self.W, self.V, self.b, self.c]]
        self.__build__()

    def __build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c
        
        x = T.ivector('x')
        y = T.ivector('y')
        
        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
            
            # Word embedding layer
            x_e = E[:,x_t]
            
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            
            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
            
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

            return [o_t, s_t1, s_t2]
        
        [o, s, s2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None, 
                          dict(initial=T.zeros(self.hidden_dim, dtype=np.float64)),
                          dict(initial=T.zeros(self.hidden_dim, dtype=np.float64))])

        self.forward = theano.function([x], o)
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Total cost (could add regularization here)
        cost = o_error
        
        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        
        # Assign functions
        self.predict = theano.function([x], o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])
        
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2
        
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

    def gradient_check(model, x, y, h=0.001, error_threshold=0.01):
        # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
        model.bptt_truncate = 1000
        # Calculate the gradients using backprop
        bptt_gradients = model.bptt(x, y)
        # List of all parameters we want to chec.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter_T = operator.attrgetter(pname)(model)
            parameter = parameter_T.get_value()
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                parameter_T.set_value(parameter)
                gradplus = model.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                parameter_T.set_value(parameter)
                gradminus = model.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                parameter[ix] = original_value
                parameter_T.set_value(parameter)
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return 
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)

    def fit(self, X_train, y_train, learning_rate=0.005, nepoch=1, nstep=-1, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        if nstep == -1: nstep = len(y_train)
        lastlost = np.inf
        num_examples_seen = 0
        for epoch in range(nepoch):
            sampleindex = np.random.randint(0, len(y_train), size=nstep)
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train[sampleindex], y_train[sampleindex])
                
                print "="*100
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                sys.stdout.flush()
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                for i in xrange(100):
                    try:
                        print "Generate[{}]: ".format(i), ' '.join(self.generate(self.dictionary))
                    except:
                        pass
                    sys.stdout.flush()

                # Adjust the learning rate if loss increases
                learning_rate = learning_rate * 0.9 if lastlost < loss else learning_rate * 1.01
                lastlost = loss
                sys.stdout.flush()
                print "Setting learning rate to %f" % learning_rate

                # Saving model oarameters
                self.save("./models/rnn-theano-%d-%d-%s.npz" % (self.hidden_dim, self.word_dim, time))

            for i in sampleindex:
                print '%d examples learnt\r' % num_examples_seen,
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
                sys.stdout.flush()

    def save(model, outfile):
        np.savez(outfile, **dict([(name, sharedvalue.get_value()) for name, sharedvalue in model.params]))
        print "Saved model parameters to %s." % outfile
       
    def load(model, path):
        npzfile = np.load(path)
        for name, sharedvalue in model.params:
            sharedvalue.set_value(npzfile[name])
        print "Loaded model parameters from %s." % (path)

    def generate(model, index_to_word, maxlen=30):
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        # We start the sentence with the start token
        new_sentence = [word_to_index[sentence_start_token]]
        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[sentence_end_token]\
                and len(new_sentence) <= maxlen:
            next_word_probs = model.forward(new_sentence)
            sampled_word = word_to_index[unknown_token]
            # We don't want to sample unknown words
            while sampled_word == word_to_index[unknown_token]:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [index_to_word[x] for x in new_sentence]
        return sentence_str


