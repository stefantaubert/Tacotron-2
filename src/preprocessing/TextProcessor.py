import numpy as np

from src.preprocessing.text.symbols import get_id, get_symbol, is_valid_text_symbol, eos_id
from src.preprocessing.text.adjustments.TextAdjuster import TextAdjuster

class TextProcessor():
  def __init__(self):
    super().__init__()

  def text_to_sequence(self, text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      Args:
        text: string to convert to a sequence

      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []
    adjuster = TextAdjuster()

    adjusted_text = adjuster.adjust(text)
    ids = _get_valid_symbolids(adjusted_text)
    sequence.extend(ids)

    # Append EOS token
    sequence.append(eos_id)

    result = np.asarray(sequence, dtype=np.int32)

    return result

  def sequence_to_text(self, sequence):
    '''Converts a sequence of IDs back to a string'''
    symbols = [get_symbol(s_id) for s_id in sequence]
    result = ''.join(symbols)

    return result

def _get_valid_symbolids(symbols):
  return [get_id(symbol) for symbol in symbols if is_valid_text_symbol(symbol)]

if __name__ == "__main__":
  proc = TextProcessor()
  inp = "hello my name is mr. test"
  outp = proc.text_to_sequence(inp)

  print(outp)
  outp_to_text = proc.sequence_to_text(outp)
  print(outp_to_text)