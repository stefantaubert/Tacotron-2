import tensorflow as tf
from src.tac.core.tacotron.models.modules.helpers import *
from src.tac.core.tacotron.models.modules.ZoneoutLSTMCell import ZoneoutLSTMCell

class DecoderRNN:
	"""Decoder two uni directional LSTM Cells
	"""
	def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is in training or inference to control zoneout
			layers: integer, the number of LSTM layers in the decoder
			size: integer, the number of LSTM units in each layer
			zoneout: the zoneout factor
		"""
		super(DecoderRNN, self).__init__()
		self.is_training = is_training

		self.layers = layers
		self.size = size
		self.zoneout = zoneout
		self.scope = 'decoder_rnn' if scope is None else scope

		#Create a set of LSTM layers
		self.rnn_layers = [ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='decoder_LSTM_{}'.format(i+1)) for i in range(layers)]

		self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)

	def __call__(self, inputs, states):
		with tf.variable_scope(self.scope):
			return self._cell(inputs, states)

