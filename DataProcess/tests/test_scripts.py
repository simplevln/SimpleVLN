import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def test_dagger_script_resolves_dataset_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            fake = path / 'python with spaces'
            fake.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
Path(os.environ['CAPTURE']).write_text(json.dumps({'argv':sys.argv[1:], 'root':os.environ['VLN_DATA_ROOT']}))
sys.exit(7)
''')
            fake.chmod(0o755)
            env = {**os.environ, 'PYTHON':str(fake), 'CAPTURE':str(path/'capture'),
                   'VLN_DATA_ROOT':str(path/'dataset'), 'OUTPUT_ROOT':str(path/'output'),
                   'DAGGER_DATASET':'RxR', 'MAX_ACTION_HISTORY':'200'}
            result = subprocess.run(['bash',str(ROOT/'scripts/dagger_simplevln.sh')],
                                    cwd=folder,env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,7,result.stderr)
            call=json.loads((path/'capture').read_text()); argv=call['argv']
            self.assertEqual(argv[argv.index('--dagger_dataset')+1],'RxR')
            self.assertEqual(argv[argv.index('--max-action-history')+1],'200')
            self.assertEqual(argv[argv.index('--dagger_data_path')+1],
                             str(path/'dataset/eval_data/rxr/train/train_guide_en.json.gz'))
            self.assertEqual(call['root'],env['VLN_DATA_ROOT'])
            self.assertNotIn('--dagger_commit_freq', argv)


if __name__ == '__main__': unittest.main()
