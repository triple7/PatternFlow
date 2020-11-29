from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Reshape, BatchNormalization, Layer, LeakyReLU, Dropout, UpSampling2D,  Flatten, Dense
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from time import time
from tensorflow.keras.initializers import RandomNormal
from reflectionPadding import ReflectionPadding1D, ReflectionPadding2D
from spectralNormalisation import SpectralNorm

def generate_discriminator(input):
	d1 = Sequential()
	d1.add(ReflectionPadding1D((7, 7)))
	d1.add(SpectralNorm()Conv1D(256, ))