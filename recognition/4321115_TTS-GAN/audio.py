from pydub import AudioSegment
from os import makedir
import speech_recognition as SR

def chop_audio(file, frame_rate=26000):
	source = AudioSegment.from_file('./data/'+file, format='wav')
	source = source.set_frame_rate(frame_rate)
	for segment in time_codes:
		start, duration = segment['start'], segment['duration']
		end = start+duration
		