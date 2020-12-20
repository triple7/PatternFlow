from tensorflow  import signal
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Multiply, Layer, Lambda
from tensorflow.keras.losses import MeanAbsoluteError

"""
Short term Fourrier transform
Customised to fit a variable Fourrier transform on 26k framerate audio data
"""
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

"""
Spectral convergence
Custom loss function for computing distances between estimated and true spectral bins
""""""
def spectral_convergence(y_true, y_pred):
	norm_reg = Subtract([y_pred,y_true])
	X = tf.norm(norm_reg)
	Y = tf.norm(y_pred)
	return Lambda(lambda X: X[0]/X[1])([X, Y])

"""
Get the log magnitude for the spectral bin
"""
def log_stft_magnitude(y_true, y_pred):
	pred_mag = K.log(y_pred)
	true_mag = K.log(y_true)
	return MeanAbsoluteError(pred_mag, true_mag)

"""
Single Short Term Fourrier transform loss
"""
def stft_loss(y_true, y_pred, fft_size, hop_size, win_size):
	pred_mag = STFT(y_pred, fft_size, hop_size, win_size)
	true_mag = STFT(y_true, fft_size, hop_size, win_size)
	
	spec_conv = spectral_convergence(pred_mag, true_mag)
	mag_loss = log_stft_magnitude(y_pred, y_true)
	
	return spec_conv, mag_loss

"""
Multi resolution spectral loss
"""
def multi_res_stft_loss(y_pred, y_true):
	spec_lossss, imag_lossss = [], []
	fft_sizes = [1024, 2048, 512]
	win_sizes = [600, 1200, 240]
	hop_sizes = [120, 240, 50]
	for i in range(len(fft_sizes)):
		spec_loss, mag_loss = stft_loss(y_pred, y_true, fft_sizes[i], hop_sizes[i], win_sizes[i])
		spec_losses.append(spec_loss)
		imag_losses.append(mag_loss)
	spec_loss = sum(spec_losses)/len(spec_losses)
	mag_loss = sum(imag_losses)/len(imag_losses)
	
	return spec_loss, mag_loss

