import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm
from hparams import hparams

from datasets import audio
from src.preprocessing.parser.LJSpeechDatasetParser import LJSpeechDatasetParser
from wavenet_vocoder.util import (is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize)
from src.preprocessing.audio.utterance_processor import process_utterance

def get_wav_dir(caching_dir: str) -> str:
  ''' The directory to write the preprocessed wav into. '''
  return os.path.join(caching_dir, 'audio')

def get_mel_dir(caching_dir: str) -> str:
  ''' The directory to write the mel spectograms into. '''
  return os.path.join(caching_dir, 'mels')

def get_lin_dir(caching_dir: str) -> str:
  ''' The directory to write the linear spectrograms into. '''
  return os.path.join(caching_dir, 'linear')

class WavProcessor():
  def __init__(self, hp: hparams, caching_dir: str):
    self.hp = hp
    self._set_paths(caching_dir)
    self._ensure_folders_exist()

  def process(self, dataset: LJSpeechDatasetParser, n_jobs):
    utterances = dataset.parse()
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    for basename, text, wav in utterances:
        process = partial(process_utterance, self.mel_dir, self.linear_dir, self.wav_dir, basename, wav, self.hp)
        x = executor.submit(process)
        futures.append(x)

    self.processing_result = [future.result() for future in tqdm(futures) if future.result() is not None]
    return self.processing_result

  def show_stats(self):
    assert self.processing_result

    timesteps_sum = sum([int(m[1]) for m in self.processing_result])
    timesteps_max = max([int(m[1]) for m in self.processing_result])
    mel_frames_sum = sum([int(m[2]) for m in self.processing_result])
    mel_frames_max = max([int(m[2]) for m in self.processing_result])
    hours = timesteps_sum / self.hp.sample_rate / 3600

    print('Written {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(len(self.processing_result), mel_frames_sum, timesteps_sum, hours))
    #print('Max input length (text chars): {}'.format(max(len(m[5]) for m in self.processing_result)))
    print('Max audio timesteps length: {}'.format(timesteps_max))
    print('Max mel frames length: {}'.format(mel_frames_max))

  def _set_paths(self, caching_dir: str):
    self.caching_dir = caching_dir
    self.mel_dir = get_mel_dir(caching_dir)
    self.wav_dir = get_wav_dir(caching_dir)
    self.linear_dir = get_lin_dir(caching_dir)

  def _ensure_folders_exist(self):
    os.makedirs(self.caching_dir, exist_ok=True)
    os.makedirs(self.mel_dir, exist_ok=True)
    os.makedirs(self.wav_dir, exist_ok=True)
    os.makedirs(self.linear_dir, exist_ok=True)
