import argparse
import os
from warnings import warn
from time import sleep

import tensorflow as tf

from hparams import hparams
from infolog import log
from wavenet_vocoder.synthesize import wavenet_synthesize

def synthesize(args, hparams, taco_checkpoint, wave_checkpoint, sentences):
	log('Synthesizing audio from mel-spectrograms.. (This may take a while)')
	wavenet_synthesize(args, hparams, wave_checkpoint)
	log('Tacotron-2 TTS synthesis complete!')

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('--caching_dir', default='/datasets/models/tacotron/cache')
  #parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  #parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
  #parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
  #parser.add_argument('--wavenet_name', help='Name of logging directory of WaveNet. If trained separately')
  #parser.add_argument('--model', default='Tacotron-2')
  #parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
  #parser.add_argument('--mels_dir', default='tacotron_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
  #parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
  accepted_modes = ['eval', 'synthesis', 'live']
  parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
  parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
  parser.add_argument('--text_list', default='', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
  parser.add_argument('--speaker_id', default=None, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')
  args = parser.parse_args()

  if args.mode not in accepted_modes:
    raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

  if args.GTA not in ('True', 'False'):
    raise ValueError('GTA option must be either True or False')

  modified_hp = hparams.parse(args.hparams)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  from src.training.WavTraining import get_log_dir
  from wavenet_vocoder.train import get_save_dir
  wavenet_log_dir = get_log_dir(args.caching_dir)
  wavenet_pretrained = get_save_dir(wavenet_log_dir)

  wavenet_synthesize(args, hparams, wavenet_pretrained)

if __name__ == '__main__':
  run()
