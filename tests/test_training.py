import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from llm_course.data import Counts,Tokenizer,prepare,windows
from llm_course.model import Config,Decoder
from llm_course.train import evaluate,load_checkpoint,parser,run,seed_everything
class DataTests(unittest.TestCase):
 def test_group_disjoint_and_windows(self):
  splits,tok,_=prepare('toy',42); self.assertEqual(splits,prepare('toy',42)[0]); a,b,c=map(set,splits.values())
  self.assertFalse(a&b or a&c or b&c)
  for groups in splits.values(): self.assertEqual(sum(len(y) for _,y in windows(groups,tok,7)),sum(map(len,groups)))
 def test_real_split_before_vocab(self):
  with tempfile.TemporaryDirectory() as temp:
   with self.assertRaisesRegex(FileNotFoundError,'不会用 toy'): prepare('real',data_dir=temp)
   Path(temp,'real.txt').write_text('a'*80+'b'*10+'c'*10)
   splits,tok,_=prepare('real',data_dir=temp); self.assertEqual(splits['train'],['a'*80])
   self.assertEqual(tok.encode('bc'),[0,0]); self.assertNotIn('b',tok.tokens)
   self.assertEqual(sum(len(y) for _,y in windows(splits['test'],tok,4)),10)
 def test_baseline_denominator(self):
  tok=Tokenizer('ABC'); baseline=Counts(['AB','AB','AC'],tok)
  rows=[([tok.index['A']],[tok.index['B']])]; score=baseline.score(rows,2)
  self.assertAlmostEqual(score['ce'],-math.log(3/8)); self.assertEqual(score['target_count'],1)
  self.assertEqual(baseline.score(rows*3,2)['target_count'],3)
 def test_model_token_denominator(self):
  seed_everything(3); model=Decoder(Config(5,8,2,1,8))
  with torch.no_grad():
   for p in model.parameters(): p.zero_()
  score=evaluate(model,[([2],[3]),([2,3,4],[3,4,1])]); self.assertEqual(score['target_count'],4)
  self.assertAlmostEqual(score['ce'],math.log(5),places=6)
class TrainingTests(unittest.TestCase):
 def test_selection_updates_roundtrip_reproducibility(self):
  with tempfile.TemporaryDirectory() as temp:
   args=parser().parse_args(['--steps','6','--eval-every','2','--d-model','8','--heads','2','--layers','1','--out',temp])
   report=run(args); loaded,tokens=load_checkpoint(Path(temp,'checkpoint.pt')); splits,tok,_=prepare('toy',args.seed)
   self.assertEqual(tokens,tok.tokens); self.assertEqual(report['test']['model'],evaluate(loaded,windows(splits['test'],tok,args.context)))
   self.assertEqual(report['selected_step'],min(report['history'],key=lambda x:x['validation_ce'])['step'])
   self.assertEqual({r['target_count'] for r in report['test'].values()},{report['test']['model']['target_count']})
   changes=report['parameter_tensors_changed_at_final_step']
   for prefix in ['token.','position.','blocks.0.ln1.','blocks.0.attn.','blocks.0.ln2.','blocks.0.mlp.','norm.','head.']:
    self.assertTrue(any(v for k,v in changes.items() if k.startswith(prefix)),prefix)
   args.out=str(Path(temp,'repeat'))
   again=run(args); self.assertEqual(report['history'],again['history']); self.assertEqual(report['test'],again['test'])
   self.assertTrue(Path(temp,'sample.txt').is_file()); self.assertTrue(Path(temp,'report.md').is_file())
 def test_refuses_existing_nonempty_output_before_training(self):
  with tempfile.TemporaryDirectory() as temp:
   marker=Path(temp,'report.json'); marker.write_text('preserve this')
   args=parser().parse_args(['--out',temp])
   with patch('llm_course.train.prepare') as prepare_spy:
    with self.assertRaisesRegex(ValueError,'--out'): run(args)
    prepare_spy.assert_not_called()
   self.assertEqual(marker.read_text(),'preserve this')
 def test_test_only_after_selection(self):
  with tempfile.TemporaryDirectory() as temp:
   args=parser().parse_args(['--steps','3','--eval-every','1','--d-model','8','--heads','2','--layers','1','--out',temp])
   original=evaluate; seen=[]
   def spy(model,rows): seen.append(rows); return original(model,rows)
   with patch('llm_course.train.evaluate',side_effect=spy): run(args)
   self.assertEqual(len(seen),5); self.assertTrue(all(r is seen[0] for r in seen[:-1])); self.assertIsNot(seen[-1],seen[0])
