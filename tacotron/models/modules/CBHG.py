import tensorflow as tf
from tacotron.models.modules.helpers import *
from tacotron.models.modules.HighwayNet import HighwayNet

class CBHG:
	def __init__(self, K, conv_channels, pool_size, projections, projection_kernel_size, n_highwaynet_layers, highway_units, rnn_units, bnorm, is_training, name=None):
		self.K = K
		self.conv_channels = conv_channels
		self.pool_size = pool_size

		self.projections = projections
		self.projection_kernel_size = projection_kernel_size
		self.bnorm = bnorm

		self.is_training = is_training
		self.scope = 'CBHG' if name is None else name

		self.highway_units = highway_units
		self.highwaynet_layers = [HighwayNet(highway_units, name='{}_highwaynet_{}'.format(self.scope, i+1)) for i in range(n_highwaynet_layers)]
		self._fw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_forward_RNN'.format(self.scope))
		self._bw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_backward_RNN'.format(self.scope))

	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			with tf.variable_scope('conv_bank'):
				#Convolution bank: concatenate on the last axis to stack channels from all convolutions
				#The convolution bank uses multiple different kernel sizes to have many insights of the input sequence
				#This makes one of the strengths of the CBHG block on sequences.
				conv_outputs = tf.concat(
					[conv1d(inputs, k, self.conv_channels, tf.nn.relu, self.is_training, 0., self.bnorm, 'conv1d_{}'.format(k)) for k in range(1, self.K+1)],
					axis=-1
					)

			#Maxpooling (dimension reduction, Using max instead of average helps finding "Edges" in mels)
			maxpool_output = tf.layers.max_pooling1d(
				conv_outputs,
				pool_size=self.pool_size,
				strides=1,
				padding='same')

			#Two projection layers
			proj1_output = conv1d(maxpool_output, self.projection_kernel_size, self.projections[0], tf.nn.relu, self.is_training, 0., self.bnorm, 'proj1')
			proj2_output = conv1d(proj1_output, self.projection_kernel_size, self.projections[1], lambda _: _, self.is_training, 0., self.bnorm, 'proj2')

			#Residual connection
			highway_input = proj2_output + inputs

			#Additional projection in case of dimension mismatch (for HighwayNet "residual" connection)
			if highway_input.shape[2] != self.highway_units:
				highway_input = tf.layers.dense(highway_input, self.highway_units)

			#4-layer HighwayNet
			for highwaynet in self.highwaynet_layers:
				highway_input = highwaynet(highway_input)
			rnn_input = highway_input

			#Bidirectional RNN
			outputs, states = tf.nn.bidirectional_dynamic_rnn(
				self._fw_cell,
				self._bw_cell,
				rnn_input,
				sequence_length=input_lengths,
				dtype=tf.float32)
			return tf.concat(outputs, axis=2) #Concat forward and backward outputs
