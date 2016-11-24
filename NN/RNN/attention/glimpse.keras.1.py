import sys
sys.path.insert(0, '/home/shaofan/.local/lib/python2.7/site-packages')
import keras
import numpy as np

keras.backend.theano_backend._set_device('dev1')
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Activation, Dropout, TimeDistributed, merge, Lambda, LSTM
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical

from keras.optimizers import Adadelta, SGD, RMSprop


