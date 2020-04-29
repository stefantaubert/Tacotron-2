from src.preprocessing.parser.LJSpeechDatasetParser import LJSpeechDatasetParser
import eng_to_ipa as ipa
from tqdm import tqdm

def convert_utterances(dataset: LJSpeechDatasetParser, destination_map: str):
  result = dataset.parse()
  with open(destination_map, 'w', encoding='utf-8') as file:
    for basename, text, _ in tqdm(result):
      ipa_text = ipa.convert(text)
      file.write('{}|{}|{}\n'.format(basename, text, ipa_text))

if __name__ == "__main__":
  dataset_path = '/datasets/LJSpeech-1.1-test'
  parser = LJSpeechDatasetParser(dataset_path)
  import os
  dest_map = os.path.join(dataset_path, 'metadata_ipa.csv')
  convert_utterances(parser, dest_map)
