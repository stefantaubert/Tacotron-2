from src.tac.core.wavenet_vocoder.models.wavenet import WaveNet
from warnings import warn
from src.tac.core.wavenet_vocoder.util import is_mulaw_quantize

def create_model(hparams, init=False):
	if is_mulaw_quantize(hparams.input_type):
		if hparams.out_channels != hparams.quantize_channels:
			raise RuntimeError(
				"out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")

	return WaveNet(hparams, init)
