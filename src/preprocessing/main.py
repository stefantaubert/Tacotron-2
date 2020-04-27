import argparse
from multiprocessing import cpu_count

from hparams import hparams
from src.preprocessing.Preprocessor import Preprocessor

if __name__ == "__main__":
  print('initializing preprocessing..')
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_path', default='/datasets/LJSpeech-1.1-test')
  parser.add_argument('--cache_path', default='/datasets/models/tacotron/cache')
  parser.add_argument('--n_jobs', type=int, default=cpu_count())
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')

  args = parser.parse_args()
  modified_hp = hparams.parse(args.hparams)

  processor = Preprocessor(args.n_jobs, args.cache_path, modified_hp)
  processor.run()
