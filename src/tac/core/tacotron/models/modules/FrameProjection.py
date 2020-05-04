import tensorflow as tf
from src.tac.core.tacotron.models.modules.helpers import *

class FrameProjection:
	"""Projection layer to r * num_mels dimensions or num_mels dimensions
	"""
	def __init__(self, shape=80, activation=None, scope=None):
		"""
		Args:
			shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
			activation: callable, activation function
			scope: FrameProjection scope.
		"""
		super(FrameProjection, self).__init__()

		self.shape = shape
		self.activation = activation

		self.scope = 'Linear_projection' if scope is None else scope
		self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			#If activation==None, this returns a simple Linear projection
			#else the projection will be passed through an activation function
			# output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
			# 	name='projection_{}'.format(self.scope))
			output = self.dense(inputs)

			return output
