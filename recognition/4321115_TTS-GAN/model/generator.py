from tensorflow import Split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Reshape, BatchNormalization, Layer, LeakyReLU, Dropout, UpSampling2D,  Flatten, Dense, Add, Multiply
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal, Orthogonal
from reflectionPadding import ReflectionPadding1D, ReflectionPadding2D
from tensorflow_addons.layers import SpectralNormalization

def conditional_batch_norm(X, n_features,):
	X = BatchNormalization()(X)
	X = Dense(n_features*2, )(X)
	X = SpectralNormalization(, kernel_initializer=RandomNormal(mean=1, stdev=0.02), bias=0)(X)
	gamma, beta = Split(X, num_or_size_splits=2, axis=1)
	gamma = gamma
	beta = beta
	X = Multiply([gamma, X])
	X = Add([X, beta])
	return X

def upsample_net(X, filters, upsample_factor):
	X = Conv1DTranspose(filters, kernel=upsampling_factor*2, strides=upsampling_factor, padding=upsample_factor//2, kernel_initializer=Orthogonal())(X)
	X = SpectralNormalization()(X)
	X = X[:, X.shape[-1]*upsample_factor]
	return X

def g_block(X, h_channels, z_channels, upsample_factor)):
	X = Conditional_batch_norm(X, h_channels)
	stack1 = Sequential()
	

def define_generator(n_channels=567, z_channels=128):
	model = Sequential()
	model.add(Conv1D(768, kernel_size=3, input_shape=(None, n_channels)))
	