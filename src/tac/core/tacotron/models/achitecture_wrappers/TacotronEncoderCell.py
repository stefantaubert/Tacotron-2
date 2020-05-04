from tensorflow.contrib.rnn import RNNCell

class TacotronEncoderCell(RNNCell):
	"""Tacotron 2 Encoder Cell
	Passes inputs through a stack of convolutional layers then through a bidirectional LSTM
	layer to predict the hidden representation vector (or memory)
	"""

	def __init__(self, convolutional_layers, lstm_layer):
		"""Initialize encoder parameters

		Args:
			convolutional_layers: Encoder convolutional block class
			lstm_layer: encoder bidirectional lstm layer class
		"""
		super(TacotronEncoderCell, self).__init__()
		#Initialize encoder layers
		self._convolutions = convolutional_layers
		self._cell = lstm_layer

	def __call__(self, inputs, input_lengths=None):
		#Pass input sequence through a stack of convolutional layers
		conv_output = self._convolutions(inputs)

		#Extract hidden representation from encoder lstm cells
		hidden_representation = self._cell(conv_output, input_lengths)

		#For shape visualization
		self.conv_output_shape = conv_output.shape
		return hidden_representation
