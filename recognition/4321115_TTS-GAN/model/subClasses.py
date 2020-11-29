from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Reshape, BatchNormalization, Layer, LeakyReLU, Dropout, UpSampling2D,  Flatten, Dense
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from time import time
from tensorflow.keras.initializers import RandomNormal

def generate_discriminator(input):
	model = Sequential()
	