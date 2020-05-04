from enum import Enum
from src.tac.preprocessing.parser.DatasetParserBase import DatasetParserBase
from src.tac.preprocessing.parser.LJSpeechDatasetParser import LJSpeechDatasetParser
from src.tac.preprocessing.parser.DummyIPADatasetParser import DummyIPADatasetParser

IPA_DUMMY = 1,
LJSPEECH = 2,
LJSPEECH_LITE = 3,
LJSPEECH_TEST = 4

def get_ipa_dummy() -> DummyIPADatasetParser:
  parser = DummyIPADatasetParser('/datasets/IPA-Dummy')
  return parser

def get_lj() -> LJSpeechDatasetParser:
  parser = LJSpeechDatasetParser('/datasets/LJSpeech-1.1')
  return parser

def get_lj_lite() -> LJSpeechDatasetParser:
  parser = LJSpeechDatasetParser('/datasets/LJSpeech-1.1-lite')
  return parser

def get_lj_test() -> LJSpeechDatasetParser:
  parser = LJSpeechDatasetParser('/datasets/LJSpeech-1.1-test')
  return parser

def get_parser(number: int) -> DatasetParserBase:
  if number == IPA_DUMMY:
    return get_ipa_dummy()
  elif number == LJSPEECH:
    return get_lj()
  elif number == LJSPEECH_LITE:
    return get_lj_lite()
  elif number == LJSPEECH_TEST:
    return get_lj_test()
  else:
    raise Exception()