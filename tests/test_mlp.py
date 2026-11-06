import unittest
import torch
from llm_course.mlp import INITIAL,TinyMLP,gradient_demo,train_demo,text_windows,forward_math

class MLPTests(unittest.TestCase):
    def test_three_gradient_methods_and_one_step(self):
        report=gradient_demo()
        self.assertTrue(all(a>0 for a in report['before']['pre_relu']))
        for actual,expected in zip(report['before']['pre_relu'],[.5,.9]):
            self.assertAlmostEqual(actual,expected)
        self.assertEqual(sum(p.numel() for p in TinyMLP().parameters()),15)
        for difference in report['max_difference'].values():
            self.assertLess(difference,1e-9)
        self.assertLess(report['after']['loss'],report['before']['loss'])
        self.assertEqual(report['initial_parameters'],INITIAL)
        for key in INITIAL:
            expected=torch.tensor(INITIAL[key])-report['learning_rate']*torch.tensor(report['analytic_gradients'][key])
            torch.testing.assert_close(torch.tensor(report['updated_parameters'][key]),expected)

    def test_math_forward_matches_torch(self):
        for x in [[1.,2.],[.2,.3],[0.,0.]]:
            expected=forward_math(INITIAL,x,2)
            logits=TinyMLP()(torch.tensor([x],dtype=torch.float64))
            torch.testing.assert_close(logits[0],torch.tensor(expected['logits'],dtype=torch.float64))

    def test_training_really_updates_and_improves(self):
        report=train_demo()
        self.assertTrue(all(report['parameter_tensors_changed'].values()))
        self.assertLess(report['train_after']['ce'],report['train_before']['ce'])
        self.assertEqual(report['evaluation']['window_count'],10)
        self.assertEqual(report['evaluation']['accuracy'],1.)
        self.assertEqual(report,train_demo())

    def test_classification_not_next_token(self):
        x,y,labels=text_windows(['AABB'])
        self.assertEqual(labels,['AA','AB','BB'])
        self.assertEqual(x.tolist(),[[0.,0.],[0.,1.],[1.,1.]])
        self.assertEqual(y.tolist(),[0,1,2])
        with self.assertRaises(ValueError): text_windows(['AX'])
        with self.assertRaises(ValueError): text_windows(['A'])
