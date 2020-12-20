from tensorflow import split
from tensorflow.keras.layers import Conv1D, , Add, Multiply, Input
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal, Orthogonal
from reflectionPadding import ReflectionPadding1D, ReflectionPadding2D
from tensorflow_addons.layers import SpectralNormalization

def conditional_batch_norm(X, n_features,):
	X = BatchNormalization()(X)
	X = Dense(n_features*2, )(X)
	X = SpectralNormalization(kernel_initializer=RandomNormal(mean=1, stdev=0.02), bias=0)(X)
	gamma, beta = split(X, num_or_size_splits=2, axis=1)
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

def g_block(X, h_channels, z_channels, upsample_factor):
	input = X
	#Conditional norm 1
	output = Conditional_batch_norm(input, h_channels)
	
	#First stack
	output = ReLU()(output)
	output = upsample_net(output, h_channels, upsample_factor)
	output = Conv1D(h_channels, kernel_size=3)(output)
	
	#Conditional batch 2
	output = Conditional_batch_norm(output, h_channels)
	
	#Second stack
	output = ReLU()(output)
	output = Conv1D(h_channels, kernel_size=3, dilation_rate=2)(output)
	
	#Residual block
	residual_output = upsample_net(input, h_channels, upsample_factor)
	residual_output = Conv1D(h_filters, kernel_size=1)(residual_output)
	residual_output = Add([residual_output, output])
	
	#Batch norm 3
	output = conditional_batch_norm(residual_output, h_channels, z_channels)
	#Third stack
	output = ReLU()(output)
	output = Conv1D(h_channels, kernel_size=3, dilation_rate=4)(output)
	
	#Batch norm 4
	output = conditional_batch_norm(output, h_channels, z_channels)
	
	#Stack four
	output = ReLU()(output)
	output = Conv1D(h_channels, kernel_size=3, dilation_rate=8)(output)
	output = Add([output, residual_output])
	return output

def define_generator(X, n_channels=567, z_channels=128):
	input_shape = X.shape
	#hard coded values for each G block
	g_block_params = [
		[768, z_channels, 1],
		[768, z_channels, 1],
		[384, z_channels, 2],
		[384, z_channels, 2],
		[384, z_channels, 2],
		[192, z_channels, 3],
		[96, z_channels, 5]
	]
	
	#Pre-processing layer
	input = Conv1D(768, kernel_size=3, input_size=n_channels)
	output = input
	#Add the G blocks
	for block in g_block_params:
		output = g_block(output, block[0], block[1], block[2])
	
	#Final generated output
	output = Conv1D(1, kernel_size=3, activation='tanh')
	return output
