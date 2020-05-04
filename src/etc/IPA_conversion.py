import eng_to_ipa as ipa

def _token_to_ipa(w, placeholder_not_found='X'):
  #has_ipa_repr = ipa.isin_cmu(w)
  ipa_result = ipa.convert(w, mode='sql')

  if '*' in ipa_result:
    w_without_punctuation = ipa.preprocess(w)
    #print("found no ipa for: {}".format(w_without_punctuation))
    placeholder = placeholder_not_found * len(w_without_punctuation)

    # because ex. "Bodoni," will not replaced without .lower()
    w = w.lower()
    
    ipa_result = w.replace(w_without_punctuation, placeholder)

  return ipa_result

def text_to_ipa(text):
  words = text.split(' ')
  ipa_text = ''

  for w in words:
    ipa_result = _token_to_ipa(w)
    ipa_text += '{} '.format(ipa_result)
  ipa_text = ipa_text.rstrip()

  return ipa_text
