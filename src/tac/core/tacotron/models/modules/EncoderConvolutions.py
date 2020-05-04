import tensorflow as tf
from src.tac.core.tacotron.models.modules.helpers import *

class EncoderConvolutions:
	"""Encoder convolutional layers used to find local dependencies in inputs characters.
	"""
	def __init__(self, is_training, hparams, activation=tf.nn.relu, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		super(EncoderConvolutions, self).__init__()
		self.is_training = is_training

		self.kernel_size = hparams.enc_conv_kernel_size
		self.channels = hparams.enc_conv_channels
		self.activation = activation
		self.scope = 'enc_conv_layers' if scope is None else scope
		self.drop_rate = hparams.tacotron_dropout_rate
		self.enc_conv_num_layers = hparams.enc_conv_num_layers
		self.bnorm = hparams.batch_norm_position

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(self.enc_conv_num_layers):
				x = conv1d(x, self.kernel_size, self.channels, self.activation, self.is_training, self.drop_rate, self.bnorm, 'conv_layer_{}_'.format(i + 1)+self.scope)
		return x
