import tensorflow as tf
from src.tac.core.tacotron.models.modules.helpers import *

class HighwayNet:
	def __init__(self, units, name=None):
		self.units = units
		self.scope = 'HighwayNet' if name is None else name

		self.H_layer = tf.layers.Dense(units=self.units, activation=tf.nn.relu, name='H')
		self.T_layer = tf.layers.Dense(units=self.units, activation=tf.nn.sigmoid, name='T', bias_initializer=tf.constant_initializer(-1.))

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			H = self.H_layer(inputs)
			T = self.T_layer(inputs)
			return H * T + inputs * (1. - T)

