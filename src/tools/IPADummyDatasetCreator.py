import os
import shutil

import eng_to_ipa as ipa
from tqdm import tqdm

from src.etc.dir_copy import copytree

from src.cmudict.src.CMUDict import get_dict
from src.tools.sentence_to_ipa import sentence_to_ipa
from src.tac.preprocessing.parser.LJSpeechDatasetParser import (
    LJSpeechDatasetParser, get_metadata_filepath)


def convert_utterances(dataset: LJSpeechDatasetParser, dest_dir: str):
  dir_exists = os.path.isdir(dest_dir) and os.path.exists(dest_dir)
  if dir_exists:
    shutil.rmtree(dest_dir)
  
  print("copying wavfiles")
  copytree(dataset.path, dest_dir)
  
  result = dataset.parse()
  dest_meta = get_metadata_filepath(dest_dir)

  cmudict = CMUDict()
  cmudict.load()
  print("loaded cmudict.")

  with open(dest_meta, 'w', encoding='utf-8') as file:
    for basename, text, _ in tqdm(result):
      ipa_text = sentence_to_ipa(text, cmudict)
      file.write('{}|{}|{}\n'.format(basename, text, ipa_text))

if __name__ == "__main__":
  import cProfile

  dataset_path = '/datasets/LJSpeech-1.1-lite'
  dest_dir = '/datasets/IPA-Dummy'

  parser = LJSpeechDatasetParser(dataset_path)
  convert_utterances(parser, dest_dir)
  #cProfile.run('convert_utterances(parser, dest_map)', None, )

  '''
     1703    9.942    0.006    9.942    0.006 {method 'execute' of 'sqlite3.Cursor' objects}
        5    0.000    0.000    0.000    0.000 {method 'extend' of 'list' objects}
     1703    7.198    0.004    7.198    0.004 {method 'fetchall' of 'sqlite3.Cursor' objects}
  '''
