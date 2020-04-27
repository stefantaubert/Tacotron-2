import collections

class CustomDecoderOutput(
		collections.namedtuple("CustomDecoderOutput", ("rnn_output", "token_output", "sample_id"))):
	pass

