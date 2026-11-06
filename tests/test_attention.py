import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('attention_lab', Path(__file__).parents[1] / 'labs/attention.py')
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

class AttentionTests(unittest.TestCase):
    def test_independent_first_row_hand_calculation(self):
        a = math.exp(1 / math.sqrt(2))
        expected = [a/(2*a+1),1/(2*a+1),a/(2*a+1)]
        r = lab.attention(lab.X)
        for actual, wanted in zip(r['weights'][0],expected):
            self.assertAlmostEqual(actual,wanted,places=12)
        self.assertAlmostEqual(r['output'][0][0],4*a/(2*a+1),places=12)
        self.assertAlmostEqual(r['output'][0][1],2*(1+a)/(2*a+1),places=12)

    def test_causal_prefix_invariance_with_changed_future(self):
        changed = [row[:] for row in lab.X]
        changed[2] = [0.0,0.0]
        before = lab.attention(lab.X,True)
        after = lab.attention(changed,True)
        self.assertEqual(before['output'][:2],after['output'][:2])
        self.assertNotEqual(lab.attention(lab.X)['output'][0],lab.attention(changed)['output'][0])
        self.assertEqual(before['weights'][0],[1,0,0])
        self.assertEqual(before['output'][0],[2,0])

    def test_normalization_and_zero_future_weights(self):
        for i,row in enumerate(lab.attention(lab.X,True)['weights']):
            self.assertAlmostEqual(sum(row),1.0)
            self.assertTrue(all(v==0 for v in row[i+1:]))

    def test_softmax_common_shift_and_large_scores(self):
        base = lab.softmax([1,2,3],[True,True,True])
        large = lab.softmax([1001,1002,1003],[True,True,True])
        for a,b in zip(base,large):self.assertAlmostEqual(a,b,places=12)

    def test_all_masked_row_is_rejected_explicitly(self):
        with self.assertRaises(ValueError):lab.softmax([1,2],[False,False])

    def test_invalid_input_is_rejected(self):
        for x in [[],[[1,2,3]],[[float('nan'),0]]]:
            with self.subTest(x=x):
                with self.assertRaises(ValueError):lab.attention(x)

if __name__ == '__main__':unittest.main()
