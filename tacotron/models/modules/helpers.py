import tensorflow as tf

def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, bnorm, scope):
	assert bnorm in ('before', 'after')
	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=activation if bnorm == 'after' else None,
			padding='same')
		batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		activated = activation(batched) if bnorm == 'before' else batched
		return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
								name='dropout_{}'.format(scope))

def _round_up_tf(x, multiple):
	# Tf version of remainder = x % multiple
	remainder = tf.mod(x, multiple)
	# Tf version of return x if remainder == 0 else x + multiple - remainder
	x_round =  tf.cond(tf.equal(remainder, tf.zeros(tf.shape(remainder), dtype=tf.int32)),
		lambda: x,
		lambda: x + multiple - remainder)

	return x_round

def sequence_mask(lengths, r, expand=True):
	'''Returns a 2-D or 3-D tensorflow sequence mask depending on the argument 'expand'
	'''
	max_len = tf.reduce_max(lengths)
	max_len = _round_up_tf(max_len, tf.convert_to_tensor(r))
	if expand:
		return tf.expand_dims(tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32), axis=-1)
	return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)

def MaskedMSE(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked Mean Squared Error
	'''

	#[batch_size, time_dimension, 1]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
	#							    [[1., 1., 1., 0., 0.]],
	#							    [[1., 1., 0., 0., 0.]]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

	#[batch_size, time_dimension, channel_dimension(mels)]
	ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
	mask_ = mask * ones

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
		return tf.losses.mean_squared_error(labels=targets, predictions=outputs, weights=mask_)

def MaskedSigmoidCrossEntropy(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked SigmoidCrossEntropy with logits
	'''

	#[batch_size, time_dimension]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[1., 0., 0., 0., 0.],
	#							    [1., 1., 1., 0., 0.],
	#							    [1., 1., 0., 0., 0.]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, False)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask))]):
		#Use a weighted sigmoid cross entropy to measure the <stop_token> loss. Set hparams.cross_entropy_pos_weight to 1
		#will have the same effect as  vanilla tf.nn.sigmoid_cross_entropy_with_logits.
		losses = tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=outputs, pos_weight=hparams.cross_entropy_pos_weight)

	with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
		masked_loss = losses * mask

	return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)

def MaskedLinearLoss(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked MAE loss with priority to low frequencies
	'''

	#[batch_size, time_dimension, 1]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
	#							    [[1., 1., 1., 0., 0.]],
	#							    [[1., 1., 0., 0., 0.]]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

	#[batch_size, time_dimension, channel_dimension(freq)]
	ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
	mask_ = mask * ones

	l1 = tf.abs(targets - outputs)
	n_priority_freq = int(2000 / (hparams.sample_rate * 0.5) * hparams.num_freq)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
		masked_l1 = l1 * mask_
		masked_l1_low = masked_l1[:,:,0:n_priority_freq]

	mean_l1 = tf.reduce_sum(masked_l1) / tf.reduce_sum(mask_)
	mean_l1_low = tf.reduce_sum(masked_l1_low) / tf.reduce_sum(mask_)

	return 0.5 * mean_l1 + 0.5 * mean_l1_low
