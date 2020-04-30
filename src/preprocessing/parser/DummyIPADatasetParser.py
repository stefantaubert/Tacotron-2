import os
from src.etc.Logger import Logger
from src.etc.IPA_symbol_extraction import extract_symbols
from src.preprocessing.parser.LJSpeechDatasetParser import LJSpeechDatasetParser
from src.preprocessing.parser.UtteranceFormat import UtteranceFormat

class DummyIPADatasetParser(LJSpeechDatasetParser):
  def __init__(self, path: str, logger: Logger = Logger()):
    super().__init__(path, logger)

  def get_format(self) -> UtteranceFormat:
    return  UtteranceFormat.IPA

if __name__ == "__main__":
  parser = DummyIPADatasetParser('/datasets/IPA-Dummy')
  result = parser.parse()
  print(result)
  