from tensorflow.keras.layers import Conv1D, LeakyReLU, 
from tensorflow_addons.layers import SpectralNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from reflectionPadding import ReflectionPadding1D, ReflectionPadding2D

"""
Defines a d6 block discriminator for use in random windowing across the input. The blocks will be in a feed forward network.
"""

def define_discriminator(X):
	#Discriminator 1 input. The (7, 7) reflective padding is over all the discriminator blocks across the input. 
	X = ReflectionPadding1D((7, 7))(X)
	X = SpectralNormalization(Conv1D(16, kernel_size=15))(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#Discriminator 2
	X = SpectralNormalization(Conv1D(64, kernel_size=41, strides=4, padding=20, groups=4))(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#Discriminator 3
	X = SpectralNormalization(Conv1D(256, kernel_size=41, strides=4, padding=20, groups=16))(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#Discriminator 4
	X = SpectralNormalization(Conv1D(1024, kernel_size=41, strides=4, padding=20, groups=64))(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#Discriminator 5
	X = SpectralNormalization(Conv1D(1024, kernel_size=41, strides=4, padding=20, groups=256))(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#Scale feature space down 
	X = SpectralNormalization(Conv1D(1024, kernel_size=5, strides=1, padding=2))(X)
	X = LeakyReLU(alpha=0.2)(X)
	#Final output
	X = SpectralNormalization(Conv1D(1, kernel_size=3, strides=1, padding=1))(X)
	return X
