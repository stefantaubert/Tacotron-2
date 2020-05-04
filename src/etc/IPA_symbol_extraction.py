from ipapy.ipastring import IPAString
from ipapy.ipachar import IPAChar, IPADiacritic

def extract_symbols(ipa: str):
  symbols = []

  for ch in ipa:
    x = IPAString(unicode_string=ch, ignore=True)
    x_len = len(x)
    was_ignored = x_len == 0

    if was_ignored:
      symbols.append(ch)
    elif x_len == 1:
      char = x[0]
      if char.is_diacritic:
        if len(symbols) > 0:
          symbols[-1] += ch
      else:
        symbols.append(ch)
    else:
      assert False
  return symbols

if __name__ == "__main__":
  y = u"ˈprɪnɪŋ, ɪn ðə ˈoʊnli sɛns wɪθ wɪʧ wi ər æt ˈprɛzənt kənˈsərnd, ˈdɪfərz frəm moʊst ɪf nɑt frəm ɔl ðə ɑrts ənd kræfts ˌrɛprɪˈzɛnɪd ɪn ðə ˌɛksəˈbɪʃən."
  #y = u"wɪʧ"
  #y = "ɪʃn̩'"
  s_ipa = IPAString(unicode_string=y, ignore=True)
  tmp = extract_symbols(y)
  print(tmp)
