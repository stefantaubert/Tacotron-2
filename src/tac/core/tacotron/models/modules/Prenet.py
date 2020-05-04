import tensorflow as tf
from src.tac.core.tacotron.models.modules.helpers import *

class Prenet:
	"""Two fully connected layers used as an information bottleneck for the attention.
	"""
	def __init__(self, is_training, layers_sizes=[256, 256], drop_rate=0.5, activation=tf.nn.relu, scope=None):
		"""
		Args:
			layers_sizes: list of integers, the length of the list represents the number of pre-net
				layers and the list values represent the layers number of units
			activation: callable, activation functions of the prenet layers.
			scope: Prenet scope.
		"""
		super(Prenet, self).__init__()
		self.drop_rate = drop_rate

		self.layers_sizes = layers_sizes
		self.activation = activation
		self.is_training = is_training

		self.scope = 'prenet' if scope is None else scope

	def __call__(self, inputs):
		x = inputs

		with tf.variable_scope(self.scope):
			for i, size in enumerate(self.layers_sizes):
				dense = tf.layers.dense(x, units=size, activation=self.activation,
					name='dense_{}'.format(i + 1))
				#The paper discussed introducing diversity in generation at inference time
				#by using a dropout of 0.5 only in prenet layers (in both training and inference).
				x = tf.layers.dropout(dense, rate=self.drop_rate, training=True,
					name='dropout_{}'.format(i + 1) + self.scope)
		return x