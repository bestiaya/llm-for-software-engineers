import unittest
import torch
from llm_course.model import Config,Decoder,generate
from llm_course.train import seed_everything
class ModelTests(unittest.TestCase):
 def setUp(self):
  seed_everything(11); self.model=Decoder(Config(9,16,4,2,12)).eval()
 def test_causality_and_cache(self):
  with torch.no_grad():
   ids=torch.tensor([[2,3,4,5],[3,2,5,4]]); full,_=self.model(ids)
   other=ids.clone(); other[:,-1]=7; result,_=self.model(other)
   torch.testing.assert_close(full[:,:-1],result[:,:-1],atol=1e-7,rtol=0)
   first,cache=self.model(ids[:,:2]); second,cache=self.model(ids[:,2:],cache)
   torch.testing.assert_close(full,torch.cat([first,second],1),atol=1e-6,rtol=1e-5)
 def test_overflow_and_stops(self):
  with torch.no_grad():
   _,cache=self.model(torch.ones((1,12),dtype=torch.long))
   with self.assertRaises(ValueError): self.model(torch.tensor([[2]]),cache)
   with self.assertRaises(ValueError): self.model(torch.ones((1,13),dtype=torch.long))
  self.assertEqual(generate(self.model,[2]*12)['stop_reason'],'context_limit')
  self.assertEqual(generate(self.model,[2],0)['stop_reason'],'token_budget')
  self.assertEqual(generate(self.model,[2,1])['stop_reason'],'eos')
 def test_gradient_finite_difference(self):
  self.model.double(); ids=torch.tensor([[2,3,4]]); self.model(ids)[0][0,-1,2].backward()
  p=self.model.blocks[0].attn.qkv.weight; index=(0,0); analytic=p.grad[index].item(); eps=1e-5
  with torch.no_grad():
   value=p[index].item(); p[index]=value+eps; plus=self.model(ids)[0][0,-1,2].item()
   p[index]=value-eps; minus=self.model(ids)[0][0,-1,2].item(); p[index]=value
  self.assertGreater(abs(analytic),1e-7); self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),places=7)
 def test_bad_config_and_cache(self):
  with self.assertRaises(ValueError): Config(9,15,4)
  self.model.train()
  with torch.no_grad(): _,cache=self.model(torch.tensor([[2]]))
  with self.assertRaises(ValueError): self.model(torch.tensor([[3]]),cache)
