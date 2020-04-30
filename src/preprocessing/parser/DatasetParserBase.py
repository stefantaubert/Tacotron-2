import os
from src.etc.Logger import Logger
from src.preprocessing.parser.UtteranceFormat import UtteranceFormat

class DatasetParserBase():
  def __init__(self, path: str, logger: Logger = Logger()):
    super().__init__()

    self.logger = logger
    self.data = None

    if not os.path.exists(path):
      print("Directory not found:", path)
      raise Exception()

    self.path = path
  
  def get_format(self) -> UtteranceFormat:
    return  UtteranceFormat.ENG

  def parse(self) -> tuple:
    ''' 
    returns tuples of each utterance string and wav filepath
    (basename, text, wav_path)
    '''
    self.logger.log("reading utterances", level=1)

    data_is_already_parsed = self.data != None

    if not data_is_already_parsed:
      result = self._parse_core()
      self.data = result

    self.logger.log("finished.", level=1)
    return self.data

  def _parse_core(self) -> tuple:
    pass