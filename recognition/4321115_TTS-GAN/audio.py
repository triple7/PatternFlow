from pydub import AudioSegment
from os import mkdir
import speech_recognition as SR

def chop_audio(file, captions, destination, transcript_destination, name, frame_rate=26100):
	source = AudioSegment.from_file(file, format='wav')
	old_frame_rate = source.frame_rate
	count = 1
	for segment in captions:
		text = segment['text']
		start, end = segment['start'], segment['end']
		clipped = source[start:end]
		clipped = clipped.set_frame_rate(frame_rate)
		clipped.export(destination+name+'_'+str(count)+'.wav', format='wav')
		with open(transcript_destination+name+'_'+str(count), 'w') as file:
			file.write(text)
			file.close()
		count += 1
