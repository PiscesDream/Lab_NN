# project adder in ideas
# non theano
import gzip
import cPickle
from itertools import product as crossproduct

from numpy import *
import matplotlib.pyplot as plt

class SOM(object):
    #rectangular
    def __init__(self, width, height):
        self.w = width
        self.h = height

    def draw(self, data, w):
        plt.clf()
        plt.plot(data[:, 0], data[:, 1], 'ro')
        for i in xrange(self.w):
            for j in xrange(self.h):
                plt.plot([w[i, j, 0]], [w[i, j, 1]], 'bx')
                if i != self.w-1:  plt.plot([w[i, j, 0], w[i+1, j, 0]], [w[i, j, 1], w[i+1, j, 1]], 'b', lw=.3)
                if j != self.h-1:  plt.plot([w[i, j, 0], w[i, j+1, 0]], [w[i, j, 1], w[i, j+1, 1]], 'b', lw=.3)
                if i != 0:  plt.plot([w[i, j, 0], w[i-1, j, 0]], [w[i, j, 1], w[i-1, j, 1]], 'b', lw=.3)
                if j != 0:  plt.plot([w[i, j, 0], w[i, j-1, 0]], [w[i, j, 1], w[i, j-1, 1]], 'b', lw=.3)
        plt.draw()


    def fit(self, data, max_iter = 10000, draw=False):
        map_radius = max(self.w, self.h)/2
        alpha_0 = .3
        alpha_d = max_iter#map_radius ** 2
        rou_0 = map_radius #max(self.w, self.h)
        rou_d = max_iter/log(map_radius) 

        n, m = data.shape
        data = data-data.min()
        data = data/data.max()
        w = random.uniform(0, 1, size=(self.w, self.h, m))
#       w_x = linspace(data[:, 0].min(), data[:, 0].max(), self.w)
#       w_y = linspace(data[:, 1].min(), data[:, 1].max(), self.h)
#       w = array(list(crossproduct(w_x, w_y))).reshape(self.w, self.h, -1)

        for t in range(max_iter):
            print 'Iteration: %d\r' % t,
            alpha = alpha_0 * exp(-t/alpha_d)
            rou = rou_0 * exp(-t/rou_d)

            vec = data[random.randint(n)]

            delta = ((w-vec)**2).sum(2)
            minind = argmin(delta)
            #maxdelta = delta.max()
            minx, miny = minind / self.w, minind % self.w

            for i in xrange(self.w):
                for j in xrange(self.h):
                    dist = (i-minx)**2+(j-miny)**2
                    if dist > rou ** 2: continue
                    #if sum((vec-w[i,j])**2) > maxdelta/(2*(1+t)/max_iter): continue
                    theta = exp(-(dist)/(2*rou**2))
                    w[i, j] = w[i, j] + theta * alpha * (vec - w[i, j]) 

            if draw:
                if t % 5000 == 0:
                    self.draw(data, w)

        self.weight = w
        raw_input('fit done')
    
    def pred(self, x):
        w = self.weight.reshape(self.w, self.h, -1)
        pred = []
        for ele in x:
            delta = ((w-ele)**2).sum(2)
            minind = argmin(delta)
            minx, miny = minind / self.w, minind % self.w
            pred.append([minx, miny])
        return array(pred)




def load_data(dataset, num = None):
    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    data_x = concatenate([train_set[0], valid_set[0], test_set[0]])
    data_y = concatenate([train_set[1], valid_set[1], test_set[1]])
    f.close()

    size = 28
    n = data_x.shape[0]
    new_data_x, new_data_y = [], []
    for _ in range(num):
        ind1 = random.randint(0, n)
        ind2 = random.randint(0, n)
        x1 = data_x[ind1]
        x2 = data_x[ind2]
        y1 = data_y[ind1]
        y2 = data_y[ind2]
        x = concatenate([x1, x2], axis=1)
        y = y1+y2
        new_data_x.append(x)
        new_data_y.append(y)

    new_data_x = array(new_data_x)
    new_data_y = array(new_data_y)
    print new_data_x.shape, new_data_y.shape

    cal = array(map(lambda x: sum(new_data_y==x), range(20)), dtype=float32)
    cal = cal/cal.sum()
    print cal 

    return new_data_x, new_data_y


def mnist_pred(som, data, label):
    plt.clf()
    Uh, Uw = 28*2, 28
    classes = 19
    pic = zeros((som.h*Uh, som.w*Uw))
    for i in range(som.w):
        for j in range(som.h):
            pic[i*Uh:(i+1)*Uh, j*Uw:(j+1)*Uw] = som.weight[i, j].reshape(Uh, Uw)
    plt.imshow(pic, cmap='binary', interpolation='none')
    plt.draw()

    pred = som.pred(data)
    vote = ones(som.weight.shape[:-1] + (classes,)) * -1
    for lab, ele in zip(label, pred):
        vote[ele[0], ele[1], lab] += 1 
    vote = vote.argmax(2)
    print vote
    pred = array(map(lambda x: vote[x[0], x[1]], pred))
    error = (pred!=label).sum()
    print 'error: %d/%d = %f' % (error, label.shape[0], float(error)/label.shape[0])

if __name__ == '__main__':
    plt.ion()
    data, label = load_data('../../Data/mnist/mnist.pkl.gz', 10000)
    print data.shape
    a = data[0]
    print a.shape
    plt.imshow(a.reshape(56, 28))
    raw_input('display')

    som = SOM(20, 20)
    print 'guessing..'
    som.fit(data, 0)
    mnist_pred(som, data, label)


    som.fit(data, 50000)

    print 'predicting..'
    mnist_pred(som, data, label)

    

    raw_input('done all')



