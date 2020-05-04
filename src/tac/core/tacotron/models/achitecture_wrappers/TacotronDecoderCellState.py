import collections

class TacotronDecoderCellState(
	collections.namedtuple("TacotronDecoderCellState",
	 ("cell_state", "attention", "time", "alignments",
	  "alignment_history", "max_attentions"))):
	"""`namedtuple` storing the state of a `TacotronDecoderCell`.
	Contains:
	  - `cell_state`: The state of the wrapped `RNNCell` at the previous time
		step.
	  - `attention`: The attention emitted at the previous time step.
	  - `time`: int32 scalar containing the current time step.
	  - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
		 emitted at the previous time step for each attention mechanism.
	  - `alignment_history`: a single or tuple of `TensorArray`(s)
		 containing alignment matrices from all time steps for each attention
		 mechanism. Call `stack()` on each to convert to a `Tensor`.
	"""
	def replace(self, **kwargs):
		"""Clones the current state while overwriting components provided by kwargs.
		"""
		return super(TacotronDecoderCellState, self)._replace(**kwargs)
