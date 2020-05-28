import unittest

from src.CMUDict.CMUDict import _get_ipa_with_stress

class UnitTests(unittest.TestCase):

  def test_UH(self):
    inp = "UH"
    res = _get_ipa_with_stress(inp)

    self.assertEqual("ʊ", res)

  def test_UH0(self):
    inp = "UH0"
    res = _get_ipa_with_stress(inp)

    self.assertEqual("ʊ", res)

  def test_UH1(self):
    inp = "UH1"
    res = _get_ipa_with_stress(inp)

    self.assertEqual("ˈʊ", res)

  def test_UH2(self):
    inp = "UH2"
    res = _get_ipa_with_stress(inp)

    self.assertEqual("ˌʊ", res)

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
