from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Reshape, BatchNormalization, Layer, LeakyReLU, Dropout, UpSampling2D,  Flatten, Dense
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from time import time
from tensorflow.keras.initializers import RandomNormal
from reflectionPadding import ReflectionPadding1D, ReflectionPadding2D
from spectralNormalisation import SpectralNormaky

def generate_discriminator(input):
	#Discriminator 1
	d1 = Sequential()
	d1.add(ReflectionPadding1D((7, 7)))
	d1.add(SpectralNorm()Conv1D(256, kernel_size=41, strides=4, padding=20, groups=16))
	d1.add(LeakyReLU(alpha=0.2))
	
	#Discriminator 2
	d2 = Sequential()
	d2.add(SpectralNorm()Conv1D(64, kernel_size=41, strides=4, padding=20, groups=4))
	d2.add(LeakyReLU(alpha=0.2))
	
	#Discriminator 3
	d3 = Sequential()
	d3.add(SpectralNorm()Conv1D(256, kernel_size=41, strides=4, padding=20, groups=16))
	d3.add(LeakyReLU(alpha=0.2))
	
	#Discriminator 4
	d4 = Sequential()
	d4.add(SpectralNorm()Conv1D(1024, kernel_size=41, strides=4, padding=20, groups=64))
	d4.add(LeakyReLU(alpha=0.2))
	
	#Discriminator 5
	d5 = Sequential()
	d5.add(SpectralNorm()Conv1D(1024, kernel_size=41, strides=4, padding=20, groups=256))
	d5.add(LeakyReLU(alpha=0.2))
	
	#Scale feature space down 
	d6 = Sequential()
	d6.add(SpectralNorm()Conv1D(1024, kernel_size=5, strides=1, padding=2))
	d6.add(LeakyReLU(alpha=0.2))
	d6.add(SpectralNorm()Conv1D(1, kernel_size=3, strides=1, padding=1))
