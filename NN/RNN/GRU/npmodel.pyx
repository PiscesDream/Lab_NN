#import numpy as np
import numpy as np
cimport numpy as np

import operator
from preprocess import load_data
from datetime import datetime
from config import *
import sys

dtype = np.float32
ctypedef np.float32_t dtype_t

cdef np.ndarray[dtype_t, ndim=2] randinit(int dim1, int dim2):
    return np.random.uniform(-np.sqrt(1./dim2), np.sqrt(1./dim2), (dim1, dim2)).astype(dtype)

cdef softmax(x):
    x = np.exp(x - np.max(x))
    return x / np.sum(x)

#cimport cython
#@cython.boundscheck(False)
class npRNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = randinit(hidden_dim, word_dim) 
        self.V = randinit(word_dim, hidden_dim)
        self.W = randinit(hidden_dim, hidden_dim)

    def forward(self, x):
        # the length of sequence
        T = len(x)
        # hidden for each time step
        cdef np.ndarray[dtype_t, ndim=2] s = np.zeros((T+1, self.hidden_dim)).astype(dtype)
        s[-1] = np.zeros(self.hidden_dim).astype(dtype)
        # output for each time step  
        cdef np.ndarray[dtype_t, ndim=2] o = np.zeros((T, self.word_dim)).astype(dtype)
        # For each time step
        cdef int t = 0
        for t from 0 <= t < T:
            # Indexing U by x[t] is the same as multiplying U with a one-hot vector x[t]
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return o, s

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward(x)

        cdef np.ndarray[dtype_t, ndim=2] dLdU = np.zeros(self.U.shape).astype(dtype)
        cdef np.ndarray[dtype_t, ndim=2] dLdV = np.zeros(self.V.shape).astype(dtype)
        cdef np.ndarray[dtype_t, ndim=2] dLdW = np.zeros(self.W.shape).astype(dtype)
        cdef np.ndarray[dtype_t, ndim=2] delta_o = o
        cdef np.ndarray[dtype_t, ndim=1] delta_t
        delta_o[np.arange(len(y)), y] -= 1.

        cdef int t
        cdef int bptt_t
        for t from T > t >= 0:
            dLdV += np.outer(delta_o[t], s[t].T)
            # init delta calc
            delta_t = self.V.T.dot(delta_o[t])*(1-(s[t]**2))
            # Back Propagation through time (at most bptt_truncate steps)
            for bptt_t from t+1 > bptt_t >= max(0, t-self.bptt_truncate):
            # for bptt_t in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_t-1])              
                dLdU[:, x[bptt_t]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1-s[bptt_t-1]**2)
        return dLdU, dLdV, dLdW


    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
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


    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        for x_i, y_i in zip(x, y):
            o, s = self.forward(x_i)
            correct_word_prediction = o[np.arange(len(y_i)), y_i]
            L += -1 * np.sum(np.log(correct_word_prediction))
        return L

    def loss(self, x, y):
        N = np.sum([len(y_i) for y_i in y])
        return self.calculate_total_loss(x, y) / N


    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def fit(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5, dictionary=None):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
        # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print '='*50
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                for i in xrange(20):
                    print "Generate[{}]: ".format(i), ' '.join(self.generate(dictionary))
            # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5 
                    print "Setting learning rate to %f" % learning_rate
                    sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
        self.save()

    def save(self, filename='models/temp.model'):
        np.savez(filename, 
                    U=self.U, 
                    V=self.V, 
                    W=self.W, 
                    wd=self.word_dim, 
                    hd=self.hidden_dim, 
                    bt=self.bptt_truncate)

    def load(self, filename='models/temp.model'):
        data = np.load(filename)
        self.U = data['U']
        self.V = data['V']
        self.W = data['W']
        self.word_dim = data['wd']
        self.hidden_dim = data['hd']
        self.bptt_truncate = data['bt']

    def generate(model, index_to_word, maxlen=20):
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        # We start the sentence with the start token
        new_sentence = [word_to_index[sentence_start_token]]
        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[sentence_end_token]\
                and len(new_sentence) <= maxlen:
            next_word_probs, _ = model.forward(new_sentence)
            sampled_word = word_to_index[unknown_token]
            # We don't want to sample unknown words
            while sampled_word == word_to_index[unknown_token]:
                try:
                    samples = np.random.multinomial(1, next_word_probs[-1])
                except:
                    pass
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [index_to_word[x] for x in new_sentence]
        return sentence_str



