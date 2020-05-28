""" from https://github.com/keithito/tacotron """

import os
import re
from src.CMUDict.CMUDictDownloader import ensure_files_are_downloaded, symbols_filename
from src.CMUDict.ARPAToIPAMapper import get_ipa_with_stress
from src.CMUDict.CMUDictParser import parse

import eng_to_ipa
import wget
from tqdm import tqdm

''' Regex for alternative pronunciation '''
_alt_re = re.compile(r'\([0-9]+\)')

class CMUDict():
  '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''

  def __init__(self):
    self._loaded = False
    
  def load(self, dictionary_dir: str = "/tmp/"):
    self._loaded = False
    paths = ensure_files_are_downloaded(dictionary_dir)
    entries = parse(paths)
    
    self._entries_arpa = entries
    self._entries_ipa = self._convert_to_ipa()
    self._entries_first_ipa = self._extract_first_ipa()
    self._loaded = True
  
  def _ensure_data_is_loaded(self):
    if not self._loaded:
      raise Exception("Please load the dictionary first.")

  def _extract_first_ipa(self):
    result = { word: ipas[0] for word, ipas in self._entries_ipa.items() }
    return result

  def _convert_to_ipa(self):
    result = { word: [] for word, _ in self._entries_arpa.items() }

    for word, pronunciations in tqdm(self._entries_arpa.items()):
      for pronunciation in pronunciations:
        phonemes = pronunciation.split(' ')
        ipa_phonemes = [get_ipa_with_stress(phoneme) for phoneme in phonemes]
        ipa = ''.join(ipa_phonemes)
        result[word].append(ipa)

    return result

  def contains(self, word: str) -> bool:
    self._ensure_data_is_loaded()
    result = word.upper() in self._entries_arpa.keys()
    return result

  def get_first_ipa(self, word: str) -> str:
    self._ensure_data_is_loaded()
    return self._entries_first_ipa[word.upper()]

  def get_all_ipa(self, word: str) -> list:
    self._ensure_data_is_loaded()
    return self._entries_ipa[word.upper()]

  def get_all_arpa(self, word):
    '''Returns list of ARPAbet pronunciations of the given word.'''
    self._ensure_data_is_loaded()
    return self._entries_arpa[word.upper()]

  def __len__(self):
    return len(self._entries_arpa)

if __name__ == "__main__":
  di = CMUDict()

  di.load()
  #print(len(di))
  print(di.get_all_arpa("test"))
  print(di.get_first_ipa("test"))
  print(di.get_all_arpa("to"))
  print(di.get_all_ipa("to"))

  # https://github.com/prosegrinder/python-cmudict Version 0.4.4. is newer than 0.7b!! has for example 'declarative' but is GPL :(
  # https://github.com/cmusphinx/cmudict Version 0.4.4.
