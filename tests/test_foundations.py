import math
import unittest
from labs.foundations import demonstration,softmax,cross_entropy,finite_difference
class FoundationsTests(unittest.TestCase):
 def test_numeric_chain(self):
  r=demonstration(); self.assertEqual(r['counting_A'],{'A':1/6,'B':3/6,'C':2/6})
  self.assertAlmostEqual(r['ce'],math.log(3)); self.assertAlmostEqual(r['updated_ce'],.9089181979565278)
  self.assertAlmostEqual(r['updated_probabilities'][1],.4029599111828766)
  for a,b in zip(r['finite_difference'],r['analytic_gradient']): self.assertAlmostEqual(a,b,places=8)
  self.assertAlmostEqual(r['chain_rule']['numeric_dL_dw'],28,places=7)
 def test_stability_and_nonuniform_gradient(self):
  self.assertEqual(softmax([1000,1000]),[.5,.5]); self.assertAlmostEqual(cross_entropy([-1000,1000],0),2000)
  z=[.5,-1.,2.]; expected=softmax(z); expected[2]-=1
  for a,b in zip(finite_difference(z,2),expected): self.assertAlmostEqual(a,b,places=8)
