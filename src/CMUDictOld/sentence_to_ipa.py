from src.CMUDict.CMUDict import CMUDict

# def _token_to_ipa(w, placeholder_not_found='X'):
#   #has_ipa_repr = ipa.isin_cmu(w)
#   ipa_result = ipa.convert(w, mode='sql')

#   if '*' in ipa_result:
#     w_without_punctuation = ipa.preprocess(w)
#     #print("found no ipa for: {}".format(w_without_punctuation))
#     placeholder = placeholder_not_found * len(w_without_punctuation)

#     # because ex. "Bodoni," will not replaced without .lower()
#     w = w.lower()
    
#     ipa_result = w.replace(w_without_punctuation, placeholder)

#   return ipa_result

punct_str = '!"#$%&\'()*+,-./:;<=>/?@[\\]^_`{|}~«»'

def remove_punct(text: str) -> str:
  result = text.strip(punct_str)

  return result

def sentence_to_ipa(text, cmudict: CMUDict, placeholder_not_found='X'):
  words = text.split(' ')
  ipa_text = ''
  ipa_words = []

  for w in words:
    w_without_punctuation = remove_punct(w)
    ipa_exists = cmudict.contains(w_without_punctuation)

    if ipa_exists:
      ipa_result = cmudict.get_first_ipa(w_without_punctuation)
    else:
      ipa_result = placeholder_not_found * len(w_without_punctuation)

    ipa_result = w.replace(w_without_punctuation, ipa_result)
    ipa_words.append(ipa_result)
  
  ipa_text =' '.join(ipa_words)

  return ipa_text

if __name__ == "__main__":
  from src.etc.IPA_symbol_extraction import extract_symbols
  dic = CMUDict()
  dic.load()
  text = "hello, this Car  \"Is a\" test yolo."
  res = sentence_to_ipa(text, dic)
  print(text)
  print(res)
  tmp = extract_symbols(res)
  print(tmp)