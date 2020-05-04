import argparse
import os
import re
import time
from time import sleep

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.tac.hparams import hparams, hparams_debug_string
from src.tac.core.tacotron.synthesizer import Synthesizer
from src.tac.infolog import log
from src.tac.preprocessing.audio.WavProcessor import (get_lin_dir, get_mel_dir,
                                                      get_wav_dir)
from src.tac.preprocessing.Preprocessor import get_train_txt, load_meta
from src.tac.preprocessing.text.conversion.SymbolConverter import get_from_file
from src.tac.preprocessing.text.TextProcessor import (get_symbols_file,
                                                      get_txt_dir)


def generate_fast(model, text):
  model.synthesize([text], None, None, None, None)

def run_live(args, checkpoint, hparams):
  # if args.mode != eval or synthesis
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

  #Log to Terminal without keeping any records in files
  log(hparams_debug_string())
  synth = Synthesizer(args.caching_dir)
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
