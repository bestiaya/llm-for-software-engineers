import tempfile
from pathlib import Path
import unittest
from llm_course.train import parser,run

class ReportingTests(unittest.TestCase):
    def test_prompt_and_generation_are_separate_and_reconstruct_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=parser().parse_args(['--steps','1','--d-model','8','--heads','2','--layers','1',
                                     '--context','16','--max-new-tokens','3','--out',str(Path(tmp)/'run')])
            report=run(args);s=report['sample']
            self.assertEqual(s['prompt_ids']+s['generated_ids'],s['ids'])
            self.assertEqual(s['prompt_decoded']+s['generated_text'],s['text'])
            self.assertEqual(s['prompt_text'],s['prompt_decoded'])
            md=(Path(tmp)/'run/report.md').read_text()
            self.assertIn('输入提示',md);self.assertIn('模型新增内容',md)

if __name__=='__main__':unittest.main()
