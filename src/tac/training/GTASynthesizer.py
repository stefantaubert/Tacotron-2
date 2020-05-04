import os
from time import sleep

import tensorflow as tf

from src.tac import infolog
from src.tac.core.tacotron.synthesize import tacotron_synthesize
from src.tac.core.tacotron.train import get_save_dir
from src.tac.hparams import hparams
from src.tac.infolog import log
from src.tac.training.TacoTrainer import get_infolog_path, get_log_dir

log = infolog.log

class GTASynthesizer:
  def __init__(self, log_dir: str, args, hp: hparams):
    super().__init__()
    self.log_dir = log_dir
    self.args = args
    self.hp = hp

  def train(self):
    log('\n#############################################################\n')
    log('Tacotron GTA Synthesis\n')
    log('###########################################################\n')
    tacotron_pretrained = get_save_dir(self.log_dir)
    input_path = tacotron_synthesize(self.args, self.hp, tacotron_pretrained)
    #tf.reset_default_graph()
    #Sleep 1/2 second to let previous graph close and avoid error messages while Wavenet is training
    #sleep(0.5)
    #input_path = os.path.join('tacotron_' + args.output_dir, 'gta', 'map.txt')


def run():
  
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--caching_dir', default='/datasets/models/tacotron/cache')
  parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
  parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')

  #parser.add_argument('--tacotron_train_steps', type=int, default=3, help='total number of tacotron training steps')
  #parser.add_argument('--checkpoint_interval', type=int, default=1, help='Steps between writing checkpoints') # 2500
  #parser.add_argument('--eval_interval', type=int, default=123, help='Steps between eval on test data')
  #parser.add_argument('--restore', type=bool, default=False, help='Set this to False to do a fresh training')
  #parser.add_argument('--summary_interval', type=int, default=1234, help='Steps between running summary ops')
  #parser.add_argument('--embedding_interval', type=int, default=123, help='Steps between updating embeddings projection visualization')

  #parser.add_argument('--base_dir', default='')
  #parser.add_argument('--tacotron_input', default='/datasets/models/tacotron/cache/train.txt')
  #parser.add_argument('--wavenet_input', default='tacotron_output/gta/map.txt')
  #parser.add_argument('--input_dir', default='/datasets/models/tacotron/cache', help='folder to contain inputs sentences/targets')
  #parser.add_argument('--name', help='Name of logging directory.')
  #parser.add_argument('--model', default='Tacotron-2')
  #parser.add_argument('--output_dir', default='output', help='folder to contain synthesized mel spectrograms')

  #parser.add_argument('--eval_interval', type=int, default=5000, help='Steps between eval on test data')
  #parser.add_argument('--tacotron_train_steps', type=int, default=100000, help='total number of tacotron training steps')
  #parser.add_argument('--wavenet_train_steps', type=int, default=500000, help='total number of wavenet training steps')
  #parser.add_argument('--wavenet_train_steps', type=int, default=3, help='total number of wavenet training steps')
  
  args = parser.parse_args()
  modified_hp = hparams.parse(args.hparams)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  log_dir = get_log_dir(args.caching_dir)
  os.makedirs(log_dir, exist_ok=True)
  infolog_path = get_infolog_path(log_dir)
  infolog.init(infolog_path, 'tacotron')

  trainer = GTASynthesizer(log_dir, args, modified_hp)
  
  trainer.train()

if __name__ == "__main__":
  run()
