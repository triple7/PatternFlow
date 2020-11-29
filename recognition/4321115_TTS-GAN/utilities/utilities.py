import webvtt as VTT
from math import floor, ceil
from audio import *

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
