import argparse
import eng_to_ipa
from random import random

class RuleBase():
  def __init__(self, likelihood=1.0):
    super().__init__()
    self.likelihood = likelihood

  def should_convert(self):
    # random() -> [0.0, 1.0)
    do_convert = self.likelihood == 1.0 or random() < self.likelihood
    return do_convert

class SentenceRule(RuleBase):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)

  def convert(self, words: list) -> list:
    if self.should_convert():
      return self._convert_core(words)
    else:
      return words
      
class WordRuleBase(RuleBase):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)

  def convert(self, words: list, current_index: int):
    if self.should_convert():
      return self._convert_core(words, current_index)
    else:
      return words[current_index].content

  def _convert_core(self, words: list, current_index: int):
    raise Exception()

class EngRule(WordRuleBase):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)

class IpaRule(WordRuleBase):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)

  def _convert_core(self, words: list):
    raise Exception()

class RuleInsertA(SentenceRule):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)

  def _convert_core(self, words: list):
    result = []
    for w in words:
      if w.content == 'test':
        word_a = InputWord('a')
        result.append(word_a)
      result.append(w)
    return result
    
class RuleHeShe(EngRule):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)

  def _convert_core(self, words: list, current_index: int):
    word = words[current_index].content
    if word == "he":
      return "she"
    elif word == "she":
      return "he"
    else:
      return word

class RuleRemoveThe(EngRule):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)

  def _convert_core(self, words: list, current_index: int):
    word = words[current_index].content
    if word == "the":
      return ""
    else:
      return word

class RuleOmit(IpaRule):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)

  def _convert_core(self, words: list, current_index: int):
    word = words[current_index].content
    if word == "fɪst":
      return "fɪs"
    elif word == "fɔːrɪst":
      return "fɔːrɪs"
    elif word == "wʊlf":
      return "wʊf"
    else:
      return word

class RuleInsert(IpaRule):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)
    
  def _convert_core(self, words: list, current_index: int):
    word = words[current_index].content
    if word == "ənd":
      return "əndə"
    elif word == "vɪlɪdʒ":
      return "vɪlɪdʒi"
    else:
      return word

class RuleSubstitue(IpaRule):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)
    
  def _convert_core(self, words: list, current_index: int):
    word = words[current_index].content
    if word == "ðer":
      return "der"
    elif word == "ðɪs":
      return "dɪs"
    else:
      return word

class RuleSubstitue2(IpaRule):
  def __init__(self, likelihood=1.0):
    super().__init__(likelihood)
    
  def _convert_core(self, words: list, current_index: int):
    word = words[current_index].content
    if word == "ðer":
      return "ser"
    elif word == "ðɪs":
      return "sɪs"
    else:
      return word

def get_relevant_rules(all_rules: list, t: type):
  relevant_rules = [r for r in all_rules if isinstance(r, t)]
  return relevant_rules

import re
r_match = "(\d+)(?:\<((?:[0]*\.\d*)|(?:0)|(?:1))\>)?"
punct_str = '!"#$%&\'()*+,-./:;<=>/?@[\\]^_`{|}~«» '

rules_dict = {
  '1': RuleOmit,
  '2': RuleInsert,
  '3': RuleSubstitue,
  '4': RuleSubstitue2,
  '5': RuleHeShe,
  '6': RuleRemoveThe,
  '7': RuleInsertA,
}

def get_rule(rule_id: str):
  result = re.match(r_match, rule_id)
  is_valid = result != None
  if is_valid:
    rule, likelihood = result.groups()
    likelihood = float(likelihood) if likelihood != None else 1.0
    if rule in rules_dict.keys():
      rule_type = rules_dict[rule]
      instance = rule_type(likelihood)
      return instance
    else:
      print('Rule {} not known.'.format(rule))
      return None
      
  print('Invalid rule'.format(rule_id))
  return None

class InputWord():
  
  def __init__(self, token: str):
    super().__init__()
    self._token = token.lower()
    self._replace_by = self._token.strip(punct_str)
    self.content = self._replace_by

  def convert_to_ipa(self):
    if not self.is_empty():
      self.content = eng_to_ipa.convert(self.content)
  
  def update(self, word):
    self.content = word

  def is_empty(self):
    return self.content == ""

  def get_result(self):
    return self._token.replace(self._replace_by, self.content)

class InputWordList():
  def __init__(self, sentence):
    super().__init__()
    self.sentence = sentence
    self._extract_words()
  
  def _extract_words(self):
    tokens = self.sentence.split(' ')
    self.input_words = [InputWord(token) for token in tokens]
  
  def transform_sentence(self, rules):
    relev_rules = get_relevant_rules(rules, SentenceRule)
    for rule in relev_rules:
      self.input_words = rule.convert(self.input_words)

  def transform_words(self, rules):
    relev_rules = get_relevant_rules(rules, EngRule)
    for rule in relev_rules:
      for i, w in enumerate(self.input_words):
        result = rule.convert(self.input_words, i)
        w.update(result)

  def convert_to_ipa(self):
    for w in self.input_words:
      w.convert_to_ipa()

  def transform_ipa(self, rules):
    relev_rules = get_relevant_rules(rules, IpaRule)
    for rule in relev_rules:
      for i, w in enumerate(self.input_words):
        result = rule.convert(self.input_words, i)
        w.update(result)

  def get_result(self):
    result = [word.get_result() for word in self.input_words if not word.is_empty()]
    return result

  def transform(self, rules):
    self.transform_sentence(rules)
    self.transform_words(rules)
    self.convert_to_ipa()
    self.transform_ipa(rules)
    result = self.get_result()
    result_str = ' '.join(result)
    return result_str

def get_rules_from_str(inp_str: str):
  rules = []
  for rule in inp_str.split(' '):
    rule_instance = get_rule(rule)
    if rule_instance != None:
      rules.append(rule_instance)
  return rules
  
def run(args):
  rules = get_rules_from_str(args.rules)
  sentences = args.text.split('.')

  for sentence in sentences:
    if len(sentence) > 0:
      wordlist = InputWordList(sentence)
      result = wordlist.transform(rules)
      print(result)

  # inp = args.text
  #inp = inp.lower()

  # # transformations on sentence level
  # inp = rules_transform(inp, rules, LANG_SENTENCE)
  # tokeized = inp.split(' ')
  # ipa_tokens = []
  # untransformed_ipa_tokens = []

  # for token in tokeized:
  #   word = token.strip(punct_str)
  #   can_mapped = eng_to_ipa.isin_cmu(word)

  #   if can_mapped:
  #     word = rules_transform(word, rules, LANG_EN)

  #     if word != "":
  #       ipa = eng_to_ipa.convert(word)
  #       untransformed_ipa_tokens.append(token.lower().replace(word.lower(), ipa))
  #       ipa = rules_transform(ipa, rules, LANG_IPA)

  #       if ipa != "":
  #         ipa_with_punct = token.lower().replace(word.lower(), ipa)
  #         ipa_tokens.append(ipa_with_punct)
  #       else:
  #         # same as next todo
  #         pass
  #     else:
  #       # todo maybe insert special chars of end to end of last word und start to start of next word
  #       pass
  #   else:
  #     print("ignore:", word)

  # print(untransformed_ipa_tokens)
  # print(ipa_tokens)
  # print(args.text)
  # print(' '.join(untransformed_ipa_tokens))
  # print(' '.join(ipa_tokens))


if __name__ == "__main__":
  x = "Printing, differs from the arts and crafts represented in the Exhibition"
  x = "fist, forest, wolf, and village There this the test."
  parser = argparse.ArgumentParser()
  parser.add_argument('-t','--text', default=x, help='input text which should be transformed')
  parser.add_argument('-r','--rules', nargs='+', default='1<1> 2<0.5> 3<.1> 4 5<0> 6 7', help='rule<likelihood>..., order plays a role')
  args = parser.parse_args()
  run(args)
  