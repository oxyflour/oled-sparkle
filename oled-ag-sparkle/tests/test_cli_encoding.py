"""Exercise CLI output with the default Windows Chinese stream encoding."""
import os
from pathlib import Path
import subprocess
import sys
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CliEncodingTest(unittest.TestCase):
    def test_help_and_demo_dry_run_under_gbk(self):
        env = dict(os.environ, PYTHONIOENCODING='gbk')
        commands = [
            [sys.executable, str(ROOT / 'scripts/sparkle_multibackend.py'), '--help'],
            [sys.executable, str(ROOT / 'scripts/run_demo.py'),
             '--backend', 'all', '--device', 'cpu', '--dry-run'],
        ]
        with tempfile.TemporaryDirectory() as directory:
            commands[1] += ['--out',directory]
            for command in commands:
                with self.subTest(command=command[1]):
                    result = subprocess.run(command, env=env, capture_output=True,
                                            text=True, encoding='gbk')
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run=json.loads((Path(directory)/'run.json').read_text(encoding='utf-8'))
            self.assertEqual(run['status'],'dry_run')
            self.assertTrue(all(item['status']=='not_run' for item in run['backend_runs'].values()))
            self.assertTrue((Path(directory)/'report.html').is_file())


if __name__ == '__main__':
    unittest.main()
