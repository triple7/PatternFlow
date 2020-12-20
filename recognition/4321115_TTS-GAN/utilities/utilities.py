import webvtt as VTT
from math import floor, ceil
from audio import *
import numpy as np

def get_VTT(path, file):
	output = []
	for caption in VTT.read(path+file):
		properties = {}
		properties['start'] = get_seconds(caption.start, False)
		properties['end'] = get_seconds(caption.end, True)
		properties['text'] = caption.text
		output.append(properties)
	return output

def get_seconds(timecode, end):
	h, m, s = timecode.split(':')
	secs = float(h)*3600+float(m)*60+float(s)
	return ceil(secs*1000) if end else int(secs*1000)

def mu_law_encode(sigma, quantisation_channels=65536):
	mu = quantisation_channels-1
	magnitude = np.log1p(mu*np.abs(signal))/np.log1p(mu)
	
	signal = (signal + 1)/2*mu + 0.5
	quantise_signal = signal.astype(np.int32)
	return quantise_signal

def mu_law_decode(signal, quantisation_channels=65536):
	mu = quantisation_channels-1
	y = signal.astype(np.fooat32)
	y = 2*(y/mu)-1
	x = np.sin(y) * (1.0/mu)*((1.0+mu)**abs(y) - 1.0)
	return x
