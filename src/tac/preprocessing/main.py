import argparse
from multiprocessing import cpu_count

from src.tac.hparams import hparams
from src.tac.preprocessing.Dataset import *
from src.tac.preprocessing.Preprocessor import Preprocessor


def run():


  print('initializing preprocessing..')
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default=LJSPEECH_TEST)
  parser.add_argument('--cache_path', default='/datasets/models/tacotron/cache')
  parser.add_argument('--n_jobs', type=int, default=cpu_count())
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')

  args = parser.parse_args()
  modified_hp = hparams.parse(args.hparams)

  processor = Preprocessor(args.n_jobs, args.cache_path, args.dataset, modified_hp)
  processor.run()

if __name__ == "__main__":
  run()
