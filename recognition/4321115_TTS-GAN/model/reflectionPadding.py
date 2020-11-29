from tensorflow import pad
from tensorflow.keras.layers import Layer

"""
1D reflection padding
"""

class ReflectionPadding1D(Layer):
	def __init__(self, padding=(1, 1), **kwargs):
		self.padding = tuple(padding)
		super(ReflectionPadding1D, self).__init__(**kwargs)
	
	def compute_output_shape(self, input_shape):
		return input_shape[1] + self.padding[0] + self.padding[1]
	
	def call(self, input_tensor, mask=None):
		padding_l, padding_r = self.padding
		return pad(input_tensor, [[0, 0], [padding_l, padding_r], [0, 0]], mode='Reflect')
	

	"""
	2D reflection padding
parameter pad: width, height padding (tuple)
	"""
	
class ReflectionPadding2D(Layer):
	def __init__(self, padding=(1, 1), **kwargs):
		self.padding = tuple(padding)
		super(ReflectionPadding2D, self).__init__(**kwargs)
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1] + 2*self.padding[0], input_shape[2] + 2*self.padding[1], input_shape[3])
	
	def call(self, input_tensor, mask=None):
		padding_w, padding_h = self.padding
		return pad(input_tensor, [[0, 0], [padding_h, padding_h], [padding_w, padding_w], [0, 0]], 'Reflect')
