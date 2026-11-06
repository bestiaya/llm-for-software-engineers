import unittest
from labs.tokenization import encode,learn_bpe
class BPETests(unittest.TestCase):
 def test_roundtrip_boundary_and_determinism(self):
  merges,rows,trace=learn_bpe(['ab ab','ab ac'],3)
  self.assertEqual(''.join(encode('ab 未知 ac',merges)),'ab 未知 ac')
  self.assertTrue(all(not any(c.isspace() for c in a+b) for a,b in merges))
  self.assertEqual((merges,rows,trace),learn_bpe(['ab ab','ab ac'],3))
  self.assertNotIn('未知',[a+b for a,b in merges])
