import numpy as np
import pickle
import os
from utilities import mu_law_encode, mu_law_decode

class DataSet:
	
	def __init__(self, path, upsampling=120):
		self.path = path
		self.metaData = self.get_metadata(path)
		self.upsampling = upsampling
	
	def __getitem__(self, index):
		sample = np.load(os.path.join(se.self.path, 'audio', self.meta_data[index]))
		condition = np.load(os.path.join(self.path, 'mel', self.meta_data[index]))
		length = min([len(sample), len(condition)*self.upsample_factor])
		
		sample = sample[length]
		condition = condition[length//self.upsample_factor]
		sample = sample.reshape(-1, 1)
		return sample, condition

	def __len__(self):
		return len(self.meta_data)
	
	def get_metadata(path):
		with open(os.path.join(path, "metadata.pkl", "rb") as file:
		return pickle.load(file))

class DataCollate:
	def __init__(self):
		upsample_factor = 120
		condition_window = 200
		self.upsample_factor = upsample_factor
		self.condition_window = condition_window
		self.sample_window = condition_window*upsample_factor
	
	def __call__(self, batch):
		return self._collate_fn_(batch)
	
def _collate_fn(self, batch):
	sample_batch = []
	condition_batch = []
	for i, x in enumerate(batch):
		if len(x[1]) < self.condition_window:
		sample = np.pad(x[0], [[0, self.sample_window - len(x[0])], [0, 0]], 'constant')
		condition = np.pad(x1], [[0, self.condition_window - len(x[1])], [0, 0]], 'edge')
	else:
		lc_index = np.random.randint(0, len(x[1]))
		sample = x[0][lc_index*self.upsample_factor:(lc_index + self.condition_window)*self.upsample_factor]
		condition = x[1][lc_index*self.upsample_factor:(lc_index + self.condition_window)*self.upsample_factor]
		sample_batch.append(sample)
		condition_batch.append(condition)
	
	sample_batch = np.stack(sample_batch)
	condition_batch = np.stack(condition_batch)
	
	sample_batch = mu_law_encode(sample_batch)
	sample_batch = mu_law_decode(sample_batch)
	