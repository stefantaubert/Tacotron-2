import tensorflow as tf
from src.tac.core.tacotron.models.modules.helpers import *
from src.tac.core.tacotron.models.modules.ZoneoutLSTMCell import ZoneoutLSTMCell

class EncoderRNN:
	"""Encoder bidirectional one layer LSTM
	"""
	def __init__(self, is_training, size=256, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control zoneout
			size: integer, the number of LSTM units for each direction
			zoneout: the zoneout factor
			scope: EncoderRNN scope.
		"""
		super(EncoderRNN, self).__init__()
		self.is_training = is_training

		self.size = size
		self.zoneout = zoneout
		self.scope = 'encoder_LSTM' if scope is None else scope

		#Create forward LSTM Cell
		self._fw_cell = ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='encoder_fw_LSTM')

		#Create backward LSTM Cell
		self._bw_cell = ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='encoder_bw_LSTM')

	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
				self._fw_cell,
				self._bw_cell,
				inputs,
				sequence_length=input_lengths,
				dtype=tf.float32,
				swap_memory=True)

			return tf.concat(outputs, axis=2) # Concat and return forward + backward outputs
