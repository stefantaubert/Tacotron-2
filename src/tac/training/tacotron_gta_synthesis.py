import argparse
import os
from time import sleep

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.tac import infolog
from src.tac.core.tacotron.synthesizer import Synthesizer
from src.tac.hparams import hparams, hparams_debug_string
from src.tac.infolog import log
from src.tac.preprocessing.audio.WavProcessor import (get_lin_dir, get_mel_dir,
                                                      get_wav_dir)
from src.tac.preprocessing.Preprocessor import get_train_txt, load_meta
from src.tac.preprocessing.text.conversion.SymbolConverter import get_from_file
from src.tac.preprocessing.text.TextProcessor import (get_symbols_file,
                                                      get_txt_dir)
from src.tac.training.tacotron_training import (get_infolog_path, get_log_dir,
                                                get_save_dir)

log = infolog.log

def get_synth_dir(caching_dir: str, gta: bool) -> str:
  ''' The directory to write the preprocessed wav into. '''
  result = ''
  output_dir = get_synthesis_output_dir(caching_dir)
  
  if gta:
    result = os.path.join(output_dir, 'gta')
  else:
    result = os.path.join(output_dir, 'natural')

  return result

def get_gta_map_file(synth_dir):
  result = os.path.join(synth_dir, 'map.txt')
  return result

def run_synthesis(args, checkpoint, hparams):

  try:
    checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
    log('loaded model at {}'.format(checkpoint_path))
  except:
    raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

  if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
    raise ValueError('Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
      hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

  if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
    raise ValueError('Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

  gta = args.GTA == 'True'
  synth_dir = get_synth_dir(args.caching_dir, gta)
  gta_map_file = get_gta_map_file(synth_dir)
  #Create output path if it doesn't exist
  os.makedirs(synth_dir, exist_ok=True)

  metadata_path = get_train_txt(args.caching_dir)
  metadata = load_meta(metadata_path)
  log(hparams_debug_string())
  synth = Synthesizer(args.caching_dir)
  synth.load(checkpoint_path, hparams, gta=gta)
  frame_shift_ms = hparams.hop_size / hparams.sample_rate
  hours = sum([int(x[2]) for x in metadata]) * frame_shift_ms / (3600)
  log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

  #Set inputs batch wise
  metadata = [metadata[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(metadata), hparams.tacotron_synthesis_batch_size)]

  log('Starting Synthesis')

  txt_dir = get_txt_dir(args.caching_dir)
  mel_dir = get_mel_dir(args.caching_dir)
  wav_dir = get_wav_dir(args.caching_dir)

  symbol_file = get_symbols_file(args.caching_dir)
  conv = get_from_file(symbol_file)
  with open(gta_map_file, 'w') as file:
    for i, meta in enumerate(tqdm(metadata)):
      text_paths = [os.path.join(txt_dir, "{}.npy".format(m[0])) for m in meta]
      text_symbols = [np.load(pth) for pth in text_paths]
      # trim ~ at the end
      texts = [conv.sequence_to_original_text(x) for x in text_symbols]
      #texts = [m[5] for m in meta]
      mel_filenames = [os.path.join(mel_dir, "{}.npy".format(m[0])) for m in meta]
      wav_filenames = [os.path.join(wav_dir, "{}.npy".format(m[0])) for m in meta]
      basenames = [m[0] for m in meta]
      mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, synth_dir, None, mel_filenames)

      for elems in zip(wav_filenames, mel_filenames, mel_output_filenames, speaker_ids, texts):
        file.write('|'.join([str(x) for x in elems]) + '\n')

  log('synthesized mel spectrograms at {}'.format(synth_dir))
  return gta_map_file

def get_synthesis_output_dir(caching_dir: str):
  return os.path.join(caching_dir, 'synthesized')

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('--caching_dir', default='/datasets/models/tacotron/cache')
  parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
  parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')

  args = parser.parse_args()
  modified_hp = hparams.parse(args.hparams)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  log_dir = get_log_dir(args.caching_dir)
  os.makedirs(log_dir, exist_ok=True)
  infolog_path = get_infolog_path(log_dir)
  infolog.init(infolog_path, 'tacotron')

  log('\n##########################################################\n')
  log('Tacotron GTA Synthesis\n')
  log('###########################################################\n')
  tacotron_pretrained = get_save_dir(log_dir)
  input_path = run_synthesis(args, tacotron_pretrained, modified_hp)
  #tf.reset_default_graph()
  #Sleep 1/2 second to let previous graph close and avoid error messages while Wavenet is training
  #sleep(0.5)

if __name__ == "__main__":
  run()
