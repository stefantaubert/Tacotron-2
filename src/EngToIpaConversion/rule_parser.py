import re

from src.EngToIpaConversion.rules.RuleHeShe import RuleHeShe
from src.EngToIpaConversion.rules.RuleInsert import RuleInsert
from src.EngToIpaConversion.rules.RuleInsertA import RuleInsertA
from src.EngToIpaConversion.rules.RuleOmit import RuleOmit
from src.EngToIpaConversion.rules.RuleRemoveThe import RuleRemoveThe
from src.EngToIpaConversion.rules.RuleSubstitue import RuleSubstitue
from src.EngToIpaConversion.rules.RuleSubstitue2 import RuleSubstitue2

_r_match = "(\d+)(?:\<((?:[0]*\.\d*)|(?:0)|(?:1))\>)?"

_rules_dict = {
  '1': RuleOmit,
  '2': RuleInsert,
  '3': RuleSubstitue,
  '4': RuleSubstitue2,
  '5': RuleHeShe,
  '6': RuleRemoveThe,
  '7': RuleInsertA,
}

def get_rule(rule_id: str):
  result = re.match(_r_match, rule_id)
  is_valid = result != None
  if is_valid:
    rule, likelihood = result.groups()
    likelihood = float(likelihood) if likelihood != None else 1.0
    if rule in _rules_dict.keys():
      rule_type = _rules_dict[rule]
      instance = rule_type(likelihood)
      return instance
    else:
      print('Rule {} unknown.'.format(rule))
      return None
      
  print('Invalid rule'.format(rule_id))
  return None

def get_rules_from_str(inp_str: str):
  rules = []
  for rule in inp_str.split(' '):
    rule_instance = get_rule(rule)
    if rule_instance != None:
      rules.append(rule_instance)
  return rules
