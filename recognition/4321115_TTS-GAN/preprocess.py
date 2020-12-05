import pickle
import os
import glob
import numpy as np
from utilities.audio import convert_audio, HOP_LENGTH, FRAME_RATE
import random
from tqdm import tqdm

TRAIN_RATE = 0.9995
TEST_RATE = 0.0005

AUDIO_DIR = './data/source/processed/'

def find_files(path, pattern='*.wav'):
	#return [f for f in glob.iglob(f'{path}/**/*/{pattern}', recursive=True)]
	return [f for f in os.listdir('./'+path)]

def prepare_data(audio_dir, mel_dir, source_path):
	mel, audio = convert_audio(AUDIO_DIR+source_path)
	np.save(audio_dir, audio, allow_pickle=False)
	np.save(mel_dir, mel, allow_pickle=False)
	return audio_dir, mel_dir, mel.shape[0]

def process(output_dir, source_files, train_dir, test_dir):
	results = []
	names = []
	random.shuffle(source_files)
	n_train = int(len(source_files)*TRAIN_RATE)
	for wave in source_files[0:n_train]:
		ID = os.path.basename(wave).replace('.wav', '.npy')
		names.append(ID)
		results.append(prepare_data(os.path.join(train_dir, "audio", ID), os.path.join(train_dir, "mel", ID), wave))
	
	with open(os.path.join(output_dir, 'test', 'names.pkl'), 'wb') as file:
		pickle.dump(names, file)
	
	names = []
	for wave in source_files[n_train:]:
		ID = os.path.basename(wave).replace('.wav', '.npy')
		names.append(ID)
		results.append(prepare_data(os.path.join(test_dir, "audio", ID), os.path.join(test_dir, "mel", ID), wave))
	
	with open(os.path.join(output_dir, "test", 'names.pkl'), 'wb') as file:
		pickle.dump(names, file)
	
	return [result for result in tqdm(results)]

def write_metaData(meta_data, output_dir):
	with open(os.path.join(output_dir, 'metaData.txt'), 'w', encoding='utf-8') as file:
		for meta in meta_data:
			file.write('|'.join([str(x) for x in meta]) + '\n')
	frames = sum([m[2] for m in meta_data])
	frame_shift_ms = HOP_LENGTH*1000 / FRAME_RATE
	hours = frames*frame_shift_ms / (3600*1000)
	print('Written %d utterances, %d frames (%.2f hours)' % (len(meta_data), frames, hours))

def pre_process(output_dir, source_files):
	print('Creating train and test audio features / mel spectrum in ./data/train and ./data/test')
	train_dir = os.path.join(output_dir, "train")
	test_dir = os.path.join(output_dir, "test")
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(train_dir, exist_ok=True)
	os.makedirs(test_dir, exist_ok=True)
	os.makedirs(os.path.join(train_dir, "audio"), exist_ok=True)
	os.makedirs(os.path.join(test_dir, "audio"), exist_ok=True)
	os.makedirs(os.path.join(train_dir, "mel"), exist_ok=True)
	os.makedirs(os.path.join(test_dir, "mel"), exist_ok=True)
	
	waves = find_files(source_files)
	meta_data = process(output_dir, waves, train_dir, test_dir)
	write_metaData(meta_data, output_dir)

if __name__ == "__main__":
	pre_process('data', 'data/source/processed')
