import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm
from hparams import hparams

from datasets import audio
from src.preprocessing.LJSpeechDatasetParser import LJSpeechDatasetParser
from wavenet_vocoder.util import (is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize)

def get_wav_dir(caching_dir: str) -> str:
  ''' The directory to write the preprocessed wav into. '''
  return os.path.join(caching_dir, 'audio')

def get_mel_dir(caching_dir: str) -> str:
  ''' The directory to write the mel spectograms into. '''
  return os.path.join(caching_dir, 'mels')

def get_lin_dir(caching_dir: str) -> str:
  ''' The directory to write the linear spectrograms into. '''
  return os.path.join(caching_dir, 'linear')

def get_train_txt(caching_dir: str) -> str:
  ''' The file that contain all preprocessed traindata metadata. '''
  return os.path.join(caching_dir, 'train.txt')

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
        process = partial(_process_utterance, self.mel_dir, self.linear_dir, self.wav_dir, basename, wav, text, self.hp)
        x = executor.submit(process)
        futures.append(x)

    self.processing_result = [future.result() for future in tqdm(futures) if future.result() is not None]

  def save_results(self):
    assert self.processing_result
    with open(self.out_train_filepath, 'w', encoding='utf-8') as f:
      for m in self.processing_result:
        line = '|'.join([str(x) for x in m]) + '\n'
        f.write(line)

  def show_stats(self):
    assert self.processing_result
    mel_frames = sum([int(m[4]) for m in self.processing_result])
    timesteps = sum([int(m[3]) for m in self.processing_result])
    sr = self.hp.sample_rate
    hours = timesteps / sr / 3600
    print('Written {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(len(self.processing_result), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[5]) for m in self.processing_result)))
    print('Max mel frames length: {}'.format(max(int(m[4]) for m in self.processing_result)))
    print('Max audio timesteps length: {}'.format(max(m[3] for m in self.processing_result)))

  def _set_paths(self, caching_dir: str):
    self.caching_dir = caching_dir
    self.mel_dir = get_mel_dir(caching_dir)
    self.wav_dir = get_wav_dir(caching_dir)
    self.linear_dir = get_lin_dir(caching_dir)
    self.out_train_filepath = get_train_txt(caching_dir)

  def _ensure_folders_exist(self):
    os.makedirs(self.caching_dir, exist_ok=True)
    os.makedirs(self.mel_dir, exist_ok=True)
    os.makedirs(self.wav_dir, exist_ok=True)
    os.makedirs(self.linear_dir, exist_ok=True)

def _process_utterance(mel_dir: str, linear_dir: str, wav_dir: str, basename: str, wav_path: str, text: str, hp: hparams):
  """
  Preprocesses a single utterance wav/text pair

  this writes the mel scale spectogram to disk and return a tuple to write
  to the train.txt file

  Args:
    - mel_dir: the directory to write the mel spectograms into
    - linear_dir: the directory to write the linear spectrograms into
    - wav_dir: the directory to write the preprocessed wav into
    - basename: the numeric index to use in the spectogram filename
    - wav_path: path to the audio file containing the speech input
    - text: text spoken in the input audio file
    - hp: hyper parameters

  Returns:
    - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
  """
  try:
    # Load the audio as numpy array
    wav = audio.load_wav(wav_path, sr=hp.sample_rate)
  except FileNotFoundError: #catch missing wav exception
    print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
      wav_path))
    return None

  #Trim lead/trail silences
  if hp.trim_silence:
    wav = audio.trim_silence(wav, hp)

  #Pre-emphasize
  preem_wav = audio.preemphasis(wav, hp.preemphasis, hp.preemphasize)

  #rescale wav
  if hp.rescale:
    wav = wav / np.abs(wav).max() * hp.rescaling_max
    preem_wav = preem_wav / np.abs(preem_wav).max() * hp.rescaling_max

    #Assert all audio is in [-1, 1]
    if (wav > 1.).any() or (wav < -1.).any():
      raise RuntimeError('wav has invalid value: {}'.format(wav_path))
    if (preem_wav > 1.).any() or (preem_wav < -1.).any():
      raise RuntimeError('wav has invalid value: {}'.format(wav_path))

  #Mu-law quantize
  if is_mulaw_quantize(hp.input_type):
    #[0, quantize_channels)
    out = mulaw_quantize(wav, hp.quantize_channels)

    #Trim silences
    start, end = audio.start_and_end_indices(out, hp.silence_threshold)
    wav = wav[start: end]
    preem_wav = preem_wav[start: end]
    out = out[start: end]

    constant_values = mulaw_quantize(0, hp.quantize_channels)
    out_dtype = np.int16

  elif is_mulaw(hp.input_type):
    #[-1, 1]
    out = mulaw(wav, hp.quantize_channels)
    constant_values = mulaw(0., hp.quantize_channels)
    out_dtype = np.float32

  else:
    #[-1, 1]
    out = wav
    constant_values = 0.
    out_dtype = np.float32

  # Compute the mel scale spectrogram from the wav
  mel_spectrogram = audio.melspectrogram(preem_wav, hp).astype(np.float32)
  mel_frames = mel_spectrogram.shape[1]

  if mel_frames > hp.max_mel_frames and hp.clip_mels_length:
    return None

  #Compute the linear scale spectrogram from the wav
  linear_spectrogram = audio.linearspectrogram(preem_wav, hp).astype(np.float32)
  linear_frames = linear_spectrogram.shape[1]

  #sanity check
  assert linear_frames == mel_frames

  if hp.use_lws:
    #Ensure time resolution adjustement between audio and mel-spectrogram
    fft_size = hp.n_fft if hp.win_size is None else hp.win_size
    l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hp))

    #Zero pad audio signal
    out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
  else:
    #Ensure time resolution adjustement between audio and mel-spectrogram
    l_pad, r_pad = audio.librosa_pad_lr(wav, hp.n_fft, audio.get_hop_size(hp), hp.wavenet_pad_sides)

    #Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
    out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

  assert len(out) >= mel_frames * audio.get_hop_size(hp)

  #time resolution adjustement
  #ensure length of raw audio is multiple of hop size so that we can use
  #transposed convolution to upsample
  out = out[:mel_frames * audio.get_hop_size(hp)]
  assert len(out) % audio.get_hop_size(hp) == 0
  time_steps = len(out)

  # Write the spectrogram and audio to disk
  audio_filename = 'audio-{}.npy'.format(basename)
  mel_filename = 'mel-{}.npy'.format(basename)
  linear_filename = 'linear-{}.npy'.format(basename)
  np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
  np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
  np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example
  return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)

