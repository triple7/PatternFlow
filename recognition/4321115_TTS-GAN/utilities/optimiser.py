from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.backend as K
import numpy as np

if K.backend() == 'tensorflow':
	import tensorflow as tf

class DecayOptimiser(Optimizer):
	
	def __init__(self, init_lr, current_step, warmup_step, decay_lr):
		super(DecayOptimiser, self).__init__(kwargs)
		self._alpha = init_lr
		with K.name_scope(self.__class__.__name__):
			self.current_step = K.Variable(current_step, dtype='int64', name='current_step')
		self.warmup_step = warmup_step
		self.decay_lr = decay_lr
	
	def get_decay_scale(self):
		if self.current_step > self.warmup_step:
			scaled_alpha = np.power(self.decay_lr, self.current_step / self.warmup_step)
		else:
			scaled_alpha = 1
		return scaled_alpha
	
	def get_updates(self, params, loss, constraints=None):
		self.current_step += 1
		lr = self.init_alpha * self.get_decay_scale()
		lr = np.maximum(10e-6, lr)
		self._alpha = lr
		params['alpha'] = self._alpha
	
	def get_config(self):
		config = {'alpha': float(K.get_value(self._alpha)),
		'current_step': int(K.get_value(self.current_step)),
		'warmup_step': self.warmup_step,
		'decay_scale': self.decay_scale
	}
		
		base_config = super(DecayConfig, self).get_config
		return dict(list(base_config.items(())) + list(config.items()))

print('ok')