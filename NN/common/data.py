import numpy as np
from toolkits import PSD

def generate_random_walk(classes, n_per_class, d_i, Time, decay=1.0):
    x = []
    y = []
    for c in range(classes):
        #  each class walks in a specific distribution
        mean = np.random.uniform(-100, 100, size=(d_i))
        cov = PSD(np.random.uniform(-100, 100, size=(d_i, d_i)))
        for i in xrange(n_per_class):
            x_i = []
            x_t = np.random.uniform(-100, 100, size=(d_i))
            d = 1.0
            for t in range(Time):
                x_i.append(x_t.reshape(d_i, 1).copy())
                addition = np.random.multivariate_normal(mean, cov)
                x_t += addition * d
                d *= decay

            x_i = np.concatenate(x_i, 1).swapaxes(0, 1)
            x.append(x_i.reshape(1, Time, d_i).copy())

        y.extend([c]*n_per_class)
    x = np.concatenate(x, axis=0).astype('float32')
    y = np.array(y).astype('int32')
    return x, y

if __name__ == '__main__':
    generate_random_walk(1,2,3,4)
