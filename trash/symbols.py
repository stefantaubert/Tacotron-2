import os

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''

# padding
_pad = '_'

# end of string
_eos = '~'

_special = '!\'\"(),-.:;? '
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

_valid_text_symbols = list(_special) + list(_characters)

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
_all_symbols = list(_pad) + list(_eos) + _valid_text_symbols

symbols_count = len(_all_symbols)

# Mappings from symbol to numeric ID and vice versa:
_id_to_symbol = {s: i for i, s in enumerate(_all_symbols)}
_symbol_to_id = {i: s for i, s in enumerate(_all_symbols)}

eos_id = _id_to_symbol[_eos]

def get_id(symbol: str) -> int:
  assert symbol in _id_to_symbol
  result = _id_to_symbol[symbol]
  return result

def get_symbol(symbol_id: int) -> str:
  assert symbol_id in _symbol_to_id
  result = _symbol_to_id[symbol_id]
  return result

def is_valid_text_symbol(symbol: str) -> bool:
  return symbol in _valid_text_symbols

def save_to_file(path: str):
  if not os.path.isfile(path):
    with open(path, 'w', encoding='utf-8') as f:
      for symbol in _all_symbols:
        if symbol == ' ':
          #For visual purposes, swap space with \s
          symbol = '\\s' 

        f.write('{}\n'.format(symbol))