import os
from time import sleep

import infolog
import tensorflow as tf
from hparams import hparams
from infolog import log
from tacotron.train import get_save_dir
from src.training.TacoTrainer import get_infolog_path
from wavenet_vocoder.train import train as wavenet_train
from tacotron.synthesize import get_synth_dir

def get_log_dir(caching_dir: str) -> str:
  ''' The directory to write the preprocessed wav into. '''
  return os.path.join(caching_dir, 'training_wavenet')

log = infolog.log

class WavTraining:
  def __init__(self, log_dir: str, args, hp: hparams):
    super().__init__()
    self.log_dir = log_dir
    self.args = args
    self.hp = hp

  def train(self):
    log('\n#############################################################\n')
    log('Wavenet Train\n')
    log('###########################################################\n')

    checkpoint = wavenet_train(log_dir, args, hparams)
    if checkpoint is None:
      raise ('Error occured while training Wavenet, Exiting!')


if __name__ == "__main__":

  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--caching_dir', default='/datasets/models/tacotron/cache')
  parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore', type=bool, default=False, help='Set this to False to do a fresh training')
  parser.add_argument('--checkpoint_interval', type=int, default=1, help='Steps between writing checkpoints') # 2500
  parser.add_argument('--eval_interval', type=int, default=123, help='Steps between eval on test data')
  parser.add_argument('--summary_interval', type=int, default=1234, help='Steps between running summary ops')
  parser.add_argument('--embedding_interval', type=int, default=123, help='Steps between updating embeddings projection visualization')
  parser.add_argument('--wavenet_train_steps', type=int, default=3, help='total number of wavenet training steps')
  
  args = parser.parse_args()
  modified_hp = hparams.parse(args.hparams)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  log_dir = get_log_dir(args.caching_dir)
  os.makedirs(log_dir, exist_ok=True)
  infolog_path = get_infolog_path(log_dir)
  infolog.init(infolog_path, 'tacotron')

  trainer = WavTraining(log_dir, args, modified_hp)
  
  trainer.train()
