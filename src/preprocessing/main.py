import argparse
import os
from multiprocessing import cpu_count

from hparams import hparams
from src.preprocessing.LJSpeechDatasetParser import LJSpeechDatasetParser
from src.preprocessing.WavProcessor import WavProcessor


def run(args, hparams):
  parser = LJSpeechDatasetParser(args.dataset_path)
  wav_processor = WavProcessor(hparams, args.cache_path)
  wav_processor.process(parser, args.n_jobs)
  wav_processor.save_results()
  wav_processor.show_stats()

if __name__ == "__main__":
  print('initializing preprocessing..')
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_path', default='/datasets/LJSpeech-1.1-test')
  parser.add_argument('--cache_path', default='/datasets/models/tacotron/cache')
  parser.add_argument('--n_jobs', type=int, default=cpu_count())
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')

  args = parser.parse_args()
  modified_hp = hparams.parse(args.hparams)

  run(args, modified_hp)
