import argparse
import os
import re
import time
from time import sleep
import numpy as np

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm
from src.preprocessing.Preprocessor import load_meta, get_train_txt
from src.preprocessing.text.TextProcessor import get_txt_dir
from src.preprocessing.audio.WavProcessor import get_mel_dir, get_lin_dir, get_wav_dir
from src.preprocessing.text.conversion.SymbolConverter import SymbolConverter
from tacotron.train import get_save_dir

def generate_fast(model, text):
  model.synthesize([text], None, None, None, None)


def run_live(args, checkpoint_path, hparams):
  #Log to Terminal without keeping any records in files
  log(hparams_debug_string())
  synth = Synthesizer()
  synth.load(checkpoint_path, hparams)

  #Generate fast greeting message
  greetings = 'Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!'
  log(greetings)
  generate_fast(synth, greetings)

  #Interaction loop
  while True:
    try:
      text = input()
      generate_fast(synth, text)

    except KeyboardInterrupt:
      leave = 'Thank you for testing our features. see you soon.'
      log(leave)
      generate_fast(synth, leave)
      sleep(2)
      break

def get_evals_dir(caching_dir):
  output_dir = get_synthesis_output_dir(caching_dir)
  eval_dir = os.path.join(output_dir, 'eval')

  return eval_dir

def run_eval(args, checkpoint_path, hparams, sentences):
  output_dir = get_synthesis_output_dir(args.caching_dir)

  eval_dir = get_evals_dir(args.caching_dir)
  log_dir = os.path.join(output_dir, 'logs-eval')

  #if args.model == 'Tacotron-2':
  #assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

  #Create output path if it doesn't exist
  os.makedirs(eval_dir, exist_ok=True)
  os.makedirs(log_dir, exist_ok=True)
  os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
  os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

  log(hparams_debug_string())
  synth = Synthesizer()
  synth.load(checkpoint_path, hparams)

  #Set inputs batch wise
  sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

  log('Starting Synthesis')
  with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
    for i, texts in enumerate(tqdm(sentences)):
      start = time.time()
      basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
      mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)

      for elems in zip(texts, mel_filenames, speaker_ids):
        file.write('|'.join([str(x) for x in elems]) + '\n')
  log('synthesized mel spectrograms at {}'.format(eval_dir))
  return eval_dir


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

def run_synthesis(args, checkpoint_path, hparams):
  gta = args.GTA == 'True'
  synth_dir = get_synth_dir(args.caching_dir, gta)
  gta_map_file = get_gta_map_file(synth_dir)
  #Create output path if it doesn't exist
  os.makedirs(synth_dir, exist_ok=True)

  metadata_path = get_train_txt(args.caching_dir)
  metadata = load_meta(metadata_path)
  log(hparams_debug_string())
  synth = Synthesizer()
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
  conv = SymbolConverter()
  with open(gta_map_file, 'w') as file:
    for i, meta in enumerate(tqdm(metadata)):
      text_paths = [os.path.join(txt_dir, "{}.npy".format(m[0])) for m in meta]
      text_symbols = [np.load(pth) for pth in text_paths]
      # trim ~ at the end
      texts = [conv.sequence_to_text(x)[:-1] for x in text_symbols]
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

def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
  #output_dir = 'tacotron_' + args.output_dir
  
  try:
    checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
    log('loaded model at {}'.format(checkpoint_path))
  except:
    raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

  if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
    raise ValueError('Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
      hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

  if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
    raise ValueError('Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
      hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

  if args.mode == 'eval':
    return run_eval(args, checkpoint_path, hparams, sentences)
  elif args.mode == 'synthesis':
    return run_synthesis(args, checkpoint_path, hparams)
  else:
    run_live(args, checkpoint_path, hparams)
