from pydub import AudioSegment
from os import mkdir
import speech_recognition as SR
import numpy as np
import scipy
import librosa


"""
General parameters for audio sampling, frequency dimension and offsets
"""

SAMPLE_RATE = 24000
n_fft = 2048
n_fft_bins = n_fft // 2+1
n_mels = 80
hop_length = 120  #frame_shift_ms * sample_rate / 1000
win_length = 240  #frame_length_ms * sample_rate / 1000
freq_min = 40
min_db = -100
ref_db = 20
mel_basis = None

"""
Pre-processing stage, take individual wav files, and take the [start:end] slices of the audio, and save them in the destination foler, associated with the transcripted text of that audio segment into its own destination folder
"""

def chop_audio(file, captions, destination, transcript_destination, name):
	source = AudioSegment.from_file(file, format='wav')
	old_frame_rate = source.frame_rate
	print("frame rate is %d" % old_frame_rate)
	count = 1
	for segment in captions:
		text = segment['text']
		start, end = segment['start'], segment['end']
		clipped = source[start:end]
		clipped = clipped.set_frame_rate(SAMPLE_RATE)
		clipped.export(destination+name+'_'+str(count)+'.wav', format='wav')
		with open(transcript_destination+name+'_'+str(count), 'w') as file:
			file.write(text)
			file.close()
		count += 1

"""
		For global parameter fixing in case the decibel level is too low for training purposes
"""

def analyse_decibels(directory, files):
	sources = [AudioSegment.from_file(directory+f, format='wav') for f in files]
	decibels = [s.dBFS for s in sources]
	average, max, min = np.mean(decibels), np.max(decibels), np.min(decibels)
	print('average %.3f max %.3f min %.3f' % (average, max, min))

def convert_audio(path):
	X = load_clip(path)
	mel = mel_spectrogram(X.astype(np.float32))
	return mel.T, x

def load_clip(path):
	return librosa.load(path, sr=FRAME_RATE)[0]

def normalise_values(spectral):
	return np.clip((spectral - min_db) / -min_db, 0, 1)

def denormalise_values(spectral):
	return (np.clip(spectral, 0, 1) * -min_db) + min_db

def amp_to_db(x):
	return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
	return np.power(10.0, x * 0.05)

def stft(y):
	return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def linear_to_mel(spectrogram):
	global mel_basis
	if mel_basis is None:
		mel_basis = generate_mel_basis()
		return np.dot(mel_basis, spectrogram)

def generate_mel_basis():
	return librosa.filters.mel(FRAME_RATE, n_fft, n_mels=n_mels, freq_min)

def mel_spectrogram(y):
	dims = stft(y)
	spectral = amp_to_db(linear_to_mel(np.abs(dims)))
	return normalise_values(spectral)

def spectrogram(y):
	dims = stft(y)
	spectral = amp_to_db(np.abs(dims)) - ref_db
	return normalise(spectral)
