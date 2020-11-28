import youtube_dl
import sys
from os import popen
from os import listdir, makedirs
#from   youtube_transcript_api   import   YouTubeTranscriptApi as transcript
from os.path import exists
from audio import *
import speech_recognition as SR

"""
Download all the files from a youtube playlist, and place them in a specific folder for audio analysis and synthesis
usage python3 loadData.py [https://youtube.com/YOUR_PLAYLIST]
"""

#The command to use for outputting youtube-dl audio
#quality 0 is best while 9 is bad.
command = 'youtube-dl -i -o "./data/%(title)s.%(ext)s" --extract-audio --audio-format wav --audio-quality 0 --yes-playlist '
#Default output folder is set to data
data_dir = 'data'

def transcribe_list(file):
	output = []
	data_files = listdir('./'+data_dir)
	n_train = len(data_files)
	with open(file, 'r') as file:
		links = file.read().split('\n')
		assert(len(links)==n_train, "Number of links doesn't match number of data source files")
		for l in links:
			dict = transcript.get_transcript(l)
			output.append(dict)
	return output, data_files

def process_audio(data_files):
	if not exists('./data_train'):
		makedir('./data_train')
	for i in range(len(transcriptions)):
		chop_sounds(transcriptions[i], data_files[i])


if __name__ == "__main__":
	print('Attempting to dowmload playlist in ./data/ as best quality wav')
	playlist = sys.argv[1]
	output = popen(command+playlist).read()
	output = output.strip('\n')
	with open('./download_output.log', 'w') as file:
		file.write(output)
	print('playlist download complete, refer to ./download_output.log for warnings/errors')
	print('Downloading individual video links')
	transcript_output = popen('youtube-dl -j --flat-playlist '+playlist+' | jq -r ".id" | sed "s_^_https://youtu.be/_" > playlist.out').read()
	with open('./playlist.out', 'w') as file:
		file.write(output.strip('\n'))
		print('Generating phrases from individual links')
		transcriptions, audio_sources = transcribe_list('playlist.out')
		print('There are %d files to transcribe, processing audio files' % len(transcriptions))
		#process_audio(audio_sources)
		print('Audio source chopping complete')
