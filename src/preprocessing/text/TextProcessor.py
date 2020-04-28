import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

from hparams import hparams
from src.preprocessing.parser.LJSpeechDatasetParser import LJSpeechDatasetParser
from src.preprocessing.text.adjustments.TextAdjuster import TextAdjuster
from src.preprocessing.text.conversion.SymbolConverter import SymbolConverter
from src.preprocessing.text.symbols import (eos_id, get_id, get_symbol, is_valid_text_symbol)

def get_txt_dir(caching_dir: str) -> str:
  ''' The directory to write the preprocessed text into. '''
  return os.path.join(caching_dir, 'preprocessing/text')

class TextProcessor():
  def __init__(self, hp: hparams, caching_dir: str):
    self.hp = hp
    self.caching_dir = caching_dir
    self._set_paths()
    self._ensure_folders_exist()
    
    self.converter = SymbolConverter()
    self.adjuster = TextAdjuster()

  def _set_paths(self):
    self.txt_dir = get_txt_dir(self.caching_dir)

  def _ensure_folders_exist(self):
    os.makedirs(self.caching_dir, exist_ok=True)
    os.makedirs(self.txt_dir, exist_ok=True)

  def process(self, dataset: LJSpeechDatasetParser, n_jobs):
    utterances = dataset.parse()
    result = []

    for basename, text, _ in tqdm(utterances):
      text = self.adjuster.adjust(text)
      # todo multicore
      sequence = self.converter.text_to_sequence(text)
      txt_filename = '{}.npy'.format(basename)
      txt_path = os.path.join(self.txt_dir, txt_filename)
      np.save(txt_path, sequence, allow_pickle=False)
      text_length = len(sequence)
      tmp = (basename, text_length)
      result.append(tmp)

    self.processing_result = result
    return result

  def show_stats(self):
    assert self.processing_result

    textlenght_sum = sum([int(m[1]) for m in self.processing_result])
    textlenght_max = max([int(m[1]) for m in self.processing_result])

    print('Written {} utterances'.format(len(self.processing_result)))
    print('Sum input length (text chars): {}'.format(textlenght_sum))
    print('Max input length (text chars): {}'.format(textlenght_max))
