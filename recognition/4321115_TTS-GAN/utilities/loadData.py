import youtube_dl
import sys
from os import popen
from os import listdir, makedirs
from   youtube_transcript_api   import   YouTubeTranscriptApi as transcript
from os.path import exists
from audio import *
from utilities import *

"""
Download all the files from a youtube playlist, and place them in a specific folder for audio analysis and synthesis
usage python3 loadData.py [https://youtube.com/YOUR_PLAYLIST]
"""

#The command to use for outputting youtube-dl audio
#quality 0 is best while 9 is bad.
#sub-titles are automatically downloaded
command = 'youtube-dl -i -o "./data/source/raw/%(title)s.%(ext)s" --extract-audio --audio-format wav --audio-quality 0 --write-auto-sub --sub-format vtt --yes-playlist '
#Default output folder is set to data
AUDIO_RAW_DIR = './data/source/raw/'
AUDIO_DIR = './data/source/processed/'
TRANSCRIPT_DIR = './data/transcript/'
LOG_DIR = './logs/'

def process_data():
	if not exists(AUDIO_DIR):
		print("creating source data directory")
		makedirs(AUDIO_DIR)
	if not exists(TRANSCRIPT_DIR):
		makedirs(TRANSCRIPT_DIR)
	data_files = [s for s in listdir(AUDIO_RAW_DIR)if '.wav' in s]
	print('Analysing sound source decibel ranges')
	analyse_decibels(AUDIO_RAW_DIR, data_files)
	data_captions = [s for s in listdir(AUDIO_RAW_DIR)if '.vtt' in s]
	data_files.sort()
	data_captions.sort()
	assert((len(data_files) == len(data_captions)), "Length of audio files is not the same as caption files")
	total_captions = 0
	for i in range(len(data_files)):
		captions = get_VTT(AUDIO_RAW_DIR, data_captions[i])
		total_captions += len(captions)
		chop_audio(AUDIO_RAW_DIR+data_files[i], captions, AUDIO_DIR, TRANSCRIPT_DIR, 'cap_'+str(i))
	print('Total segments generated %d' %total_captions)

def load_data(playlist):
	print('Attempting to dowmload playlist in ./data/source as best quality wav')
	playlist_suffix = playlist.split('?')[-1]
	if not exists(AUDIO_RAW_DIR):
		makedirs(AUDIO_RAW_DIR)
		if not exists(LOG_DIR):
			makedirs(LOG_DIR)
	output = popen(command+playlist).read()
	with open(LOG_DIR+'download_output.log', 'w') as file:
		file.write(output)
	print('playlist download complete, refer to logs/download_output.log for warnings/errors')
	print('Downloading individual video links')
	trans_command = 'youtube-dl -j --flat-playlist "'+playlist+'" | jq -r \'.id\' | sed \'s_^_https://youtube.com/watch?v=_\''
	transcript_output = popen(trans_command).read()
	with open(LOG_DIR+'playlist.out', 'w') as file:
		file.write(transcript_output.strip('\n'))
		file.close()
	print('Generating caption/audio clip pairs')
	process_data()
	print('Audio source chopping complete')

if __name__ == '__main__':
	load_data(sys.argv[1])
