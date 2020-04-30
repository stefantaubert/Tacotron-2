from src.preprocessing.parser.LJSpeechDatasetParser import LJSpeechDatasetParser
import eng_to_ipa as ipa
from tqdm import tqdm


def token_to_ipa(w, placeholder_not_found='_'):
  #has_ipa_repr = ipa.isin_cmu(w)
  ipa_result = ipa.convert(w, mode='sql')

  if '*' in ipa_result:
    w_without_punctuation = ipa.preprocess(w)
    print("found no ipa for: {}".format(w_without_punctuation))
    placeholder = '_' * len(w_without_punctuation)

    # because ex. "Bodoni," will not replaced without .lower()
    w = w.lower()
    
    ipa_result = w.replace(w_without_punctuation, placeholder)
  return ipa_result

def text_to_ipa(text, placeholder_not_found='_'):
  words = text.split(' ')
  ipa_text = ''

  for w in words:
    ipa_result = token_to_ipa(w, placeholder_not_found)
    ipa_text += '{} '.format(ipa_result)
  ipa_text = ipa_text.rstrip()

  return ipa_text

def convert_utterances(dataset: LJSpeechDatasetParser, destination_map: str):
  result = dataset.parse()
  with open(destination_map, 'w', encoding='utf-8') as file:
    for basename, text, _ in tqdm(result):
      ipa_text = text_to_ipa(text)
      file.write('{}|{}|{}\n'.format(basename, text, ipa_text))

if __name__ == "__main__":
  import cProfile
  dataset_path = '/datasets/LJSpeech-1.1-test'
  parser = LJSpeechDatasetParser(dataset_path)
  import os
  dest_map = os.path.join(dataset_path, 'metadata_ipa.csv')
  #convert_utterances(parser, dest_map)
  cProfile.run('convert_utterances(parser, dest_map)', None, )

  '''
     1703    9.942    0.006    9.942    0.006 {method 'execute' of 'sqlite3.Cursor' objects}
        5    0.000    0.000    0.000    0.000 {method 'extend' of 'list' objects}
     1703    7.198    0.004    7.198    0.004 {method 'fetchall' of 'sqlite3.Cursor' objects}
  '''