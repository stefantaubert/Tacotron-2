import os
from src.etc.Logger import Logger

class LJSpeechDatasetParser():
  def __init__(self, path: str, logger: Logger = Logger()):
    self.logger = logger
    self.data = None

    if not os.path.exists(path):
      print("Directory not found:", path)
      raise Exception()

    self.path = path

    self.metadata_filepath = self._get_metadata_filepath()

    if not os.path.exists(self.metadata_filepath):
      print("Metadatafile not found:", self.metadata_filepath)
      raise Exception()

    self.wav_dirpath = self._get_wav_dirpath()

    if not os.path.exists(self.wav_dirpath):
      print("WAVs not found:", self.wav_dirpath)
      raise Exception()

  def parse(self) -> tuple:
    ''' returns tuples of each utterance string and wav filepath '''
    data_is_already_parsed = self.data != None

    if not data_is_already_parsed:
      self.logger.log("reading utterances", level=1)
      index = 1
      result = []

      with open(self.metadata_filepath, encoding='utf-8') as f:
        for line in f:
          tmp = self._parse_line(line)
          result.append(tmp)

      self.logger.log("finished.", level=1)
      self.data = result

      self._update_characters()

    return self.data

  def _update_characters(self):
    result = set()

    for _, text, _ in self.data:
      result.update(text)
    
    self.symbols = result
    return self.symbols

  def _get_wav_dirpath(self) -> str:
    result = os.path.join(self.path, 'wavs')
    return result

  def _get_metadata_filepath(self) -> str:
    result = os.path.join(self.path, 'metadata.csv')
    return result

  def _parse_line(self, line: str) -> tuple:
    parts = line.strip().split('|')
    basename = parts[0]
    # parts[1] contains years, in parts[2] the years are written out
    # ex. ['LJ001-0045', '1469, 1470;', 'fourteen sixty-nine, fourteen seventy;']
    wav_path = os.path.join(self.wav_dirpath, '{}.wav'.format(basename))
    text = parts[2]
    tmp = (basename, text, wav_path)
    return tmp

if __name__ == "__main__":
  #parser = LJSpeechDatasetParser('/datasets/LJSpeech-1.1-lite')
  parser = LJSpeechDatasetParser('/datasets/LJSpeech-1.1')
  result = parser.parse()
  #print(result)
  chars = sorted(parser.symbols)
  print(chars)
  