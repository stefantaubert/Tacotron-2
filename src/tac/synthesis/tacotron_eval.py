import argparse
import os
import time

import tensorflow as tf
from tqdm import tqdm

from src.tac.core.tacotron.synthesizer import Synthesizer
from src.tac.hparams import hparams, hparams_debug_string
from src.tac.infolog import log


def get_evals_dir(caching_dir):
  output_dir = get_synthesis_output_dir(caching_dir)
  eval_dir = os.path.join(output_dir, 'eval')

  return eval_dir

def run_eval(args, checkpoint, hparams, sentences):
  try:
    checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
    log('loaded model at {}'.format(checkpoint_path))
  except:
    raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

  if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
    raise ValueError('Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
      hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

  if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
    raise ValueError('Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
      hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

  output_dir = get_synthesis_output_dir(args.caching_dir)

  eval_dir = get_evals_dir(args.caching_dir)
  log_dir = os.path.join(output_dir, 'logs-eval')

  #if args.model == 'Tacotron-2':
  #assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

  #Create output path if it doesn't exist
  os.makedirs(eval_dir, exist_ok=True)
  os.makedirs(log_dir, exist_ok=True)
  os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
  os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

  log(hparams_debug_string())
  synth = Synthesizer(args.caching_dir)
  synth.load(checkpoint_path, hparams)

  #Set inputs batch wise
  sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

  log('Starting Synthesis')
  with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
    for i, texts in enumerate(tqdm(sentences)):
      start = time.time()
      basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
      mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)

      for elems in zip(texts, mel_filenames, speaker_ids):
        file.write('|'.join([str(x) for x in elems]) + '\n')
  log('synthesized mel spectrograms at {}'.format(eval_dir))
  return eval_dir

def get_synthesis_output_dir(caching_dir: str):
  return os.path.join(caching_dir, 'synthesized')

def get_sentences(args):
  if args.text_list != '':
    with open(args.text_list, 'rb') as f:
      sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
  else:
    sentences = hparams.sentences
  return sentences

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('--caching_dir', default='/datasets/models/tacotron/cache')
  parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  #parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
  #parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
  #parser.add_argument('--wavenet_name', help='Name of logging directory of WaveNet. If trained separately')
  #parser.add_argument('--model', default='Tacotron-2')
  #parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
  #parser.add_argument('--mels_dir', default='tacotron_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
  #parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
  accepted_modes = ['eval', 'synthesis', 'live']
  parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
  parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
  parser.add_argument('--text_list', default='', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
  parser.add_argument('--speaker_id', default=None, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')
  args = parser.parse_args()

  if args.mode not in accepted_modes:
    raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

  if args.GTA not in ('True', 'False'):
    raise ValueError('GTA option must be either True or False')

  modified_hp = hparams.parse(args.hparams)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  from src.tac.training.tacotron_training import get_log_dir
  from src.tac.training.tacotron_training import get_save_dir
  taco_log_dir = get_log_dir(args.caching_dir)
  tacotron_pretrained = get_save_dir(taco_log_dir)

  #run_name = args.name or args.tacotron_name or args.model
  #taco_checkpoint = os.path.join('logs-' + run_name, 'taco_' + args.checkpoint)

  sentences = get_sentences(args)
  log('Synthesizing mel-spectrograms from text..')
  run_eval(args, tacotron_pretrained, modified_hp, sentences)

  #Delete Tacotron model from graph
  #tf.reset_default_graph()
  #Sleep 1/2 second to let previous graph close and avoid error messages while Wavenet is synthesizing
  #sleep(0.5)

if __name__ == "__main__":
  run()