import argparse
import os

from src.tac.hparams import hparams
from src.tac.preprocessing.audio.WavProcessor import WavProcessor
from src.tac.preprocessing.Dataset import get_parser
from src.tac.preprocessing.parser.DatasetParserBase import DatasetParserBase
from src.tac.preprocessing.text.TextProcessor import TextProcessor


def get_train_txt(caching_dir: str) -> str:
  ''' The file that contain all preprocessed traindata metadata. '''
  return os.path.join(caching_dir, 'preprocessing/train.txt')

def save_meta(data: list, path: str):
  with open(path, 'w', encoding='utf-8') as f:
    for i, tuples in enumerate(data):
      line = '|'.join([str(x) for x in tuples]) + '\n'
      f.write(line)

def load_meta(path: str) -> list:
  result = []
  with open(path, encoding='utf-8') as f:
    result = [line.strip().split('|') for line in f]
  return result

class Preprocessor():
  def __init__(self, n_jobs:int, cache_path: str, parser: int, hp: hparams):
    self.n_jobs = n_jobs
    self.out_train_filepath = get_train_txt(cache_path)
    self.text_processor = TextProcessor(hp, cache_path)
    self.wav_processor = WavProcessor(hp, cache_path)
    self.parser = get_parser(parser)

  def run(self):
    self.text_paths = self.text_processor.process(self.parser, self.n_jobs)
    self.wav_paths = self.wav_processor.process(self.parser, self.n_jobs)

    self.save_results()

    self.text_processor.show_stats()
    self.wav_processor.show_stats()

  def save_results(self):
    assert self.text_paths
    assert self.wav_paths
    
    # check no data in wav is skipped
    for i, txt in enumerate(self.text_paths):
      m = self.wav_paths[i]
      txt_name = txt[0]
      wav_name = m[0]
      assert txt_name == wav_name

    save_meta(self.wav_paths, self.out_train_filepath)

def run():
  from multiprocessing import cpu_count
  from src.tac.preprocessing.Dataset import LJSPEECH_TEST, LJSPEECH_LITE, LJSPEECH
  from src.tac.hparams import hparams

  print('initializing preprocessing..')
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default=LJSPEECH)
  parser.add_argument('--cache_path', default='/datasets/models/tacotron/cache')
  parser.add_argument('--n_jobs', type=int, default=cpu_count())
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')

  args = parser.parse_args()
  modified_hp = hparams.parse(args.hparams)

  processor = Preprocessor(args.n_jobs, args.cache_path, args.dataset, modified_hp)
  processor.run()

if __name__ == "__main__":
  run()
