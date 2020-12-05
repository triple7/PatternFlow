from tensorflow  import signal
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Multiply, Layer, Lambda
from tensorflow.keras.losses import MeanAbsoluteError

def STFT(X, fft_size, hop_size, win_size, window):
	X_spectral = signal.stft(X, frame_length=win_size, frame_step=hop_size, fft_length=fft_size,window_fn=tf.signal.hann_window)
	real = X_spectral[0]
	real = Multiply([real, real])
	imag = X_spectral[1]
	imag = Multiply([imag, imag])
	input = Add([imag, real])
	output = K.clip(input, min_value=1e-7)
	output = K.Transpose(output)
	return K.sqrt(output)

def spectral_convergence(y_true, y_pred):
	norm_reg = Subtract([y_pred,y_true])
	X = tf.norm(norm_reg)
	Y = tf.norm(y_pred)
	return Lambda(lambda X: X[0]/X[1])([X, Y])

def log_stft_magnitude(y_true, y_pred):
	pred_mag = K.log(y_pred)
	true_mag = K.log(y_true)
	return MeanAbsoluteError(pred_mag, true_mag)

def stft_loss(y_true, y_pred):
	pred_mag = STFT(y_pred, 1024, 120, 600)
	true_mag = STFT(y_true, 1024, 120, 600)
	
	spec_conv = spectral_convergence(pred_mag, true_mag)
	mag_loss = log_stft_magnitude(_pred, y_truelayer):
	
	return spec_conv, mag_loss

def multi_stft_loss(layer):
	
	def multi_res_stft_loss_fn(y_pred, y_true):
		fft_sizes = [1024, 2048, 512]
		win_sizes = [600, 1200, 240]
		hop_sizes = [120, 240, 50]
		