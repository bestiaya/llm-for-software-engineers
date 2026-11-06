import math
import unittest
import torch
from labs.evaluation import evaluate,evidence_supported,retrieve
from llm_course.experiments import demonstration,masked_ce,sft_targets
class EvaluationTests(unittest.TestCase):
 def test_retrieval_and_failure(self):
  self.assertEqual(len(evaluate()['cases']),3); self.assertFalse(evidence_supported('无限重试','timeout'))
  self.assertFalse(evidence_supported('','timeout')); self.assertFalse(evidence_supported('保留7天','missing'))
  self.assertEqual(retrieve('备份保留几天？'),['backup'])
 def test_sft_mask_and_gradient(self):
  x,y,mask=sft_targets([2,3,4,5,1],3); self.assertEqual(mask,[False,False,True,True]); self.assertEqual(y,[3,4,5,1])
  logits=torch.zeros(4,6,requires_grad=True); loss=masked_ce(logits,torch.tensor(y),torch.tensor(mask)); loss.backward()
  self.assertAlmostEqual(loss.item(),math.log(6),places=6); self.assertEqual(logits.grad[:2].abs().sum().item(),0)
  self.assertGreater(logits.grad[2:].abs().sum().item(),0)
  with self.assertRaises(ValueError): masked_ce(logits,torch.tensor(y),torch.zeros(4,dtype=torch.bool))
 def test_lora_and_cache(self):
  r=demonstration(); self.assertEqual(r['lora']['parameter_changed'],{'A':True,'B':True,'base.weight':False})
  self.assertLess(r['cache_max_error'],2e-6)
