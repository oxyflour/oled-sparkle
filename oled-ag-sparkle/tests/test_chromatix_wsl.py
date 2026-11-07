"""Exercise the real Windows-to-WSL bridge, including path and failure handling."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'win32' and os.environ.get('SPARKLE_TEST_OPTIONAL') == '1',
                     'requires Windows, WSL CUDA and SPARKLE_TEST_OPTIONAL=1')
class ChromatixWslTest(unittest.TestCase):
    def command(self, script, out):
        return [sys.executable, str(script),
                '--height', str(ROOT / 'assets/demo_quick/heightfield.png'),
                '--height-pitch-um', '2', '--height-range-um', '.5',
                '--emission', str(ROOT / 'assets/demo_quick/emission.png'),
                '--emission-pitch-um', '64', '--analysis-pitch-um', '64',
                '--roi-margin-um', '0', '--backend', 'chromatix', '--device', 'cpu',
                '--progress-every', '0', '--out', str(out)]

    def test_wsl_gpu_metadata_and_paths_with_spaces_and_unicode(self):
        with tempfile.TemporaryDirectory(prefix='chromatix bridge ') as directory:
            root = Path(directory)
            scripts = root / 'scripts with spaces 测试'
            scripts.mkdir()
            for script in (ROOT / 'scripts').glob('*.py'):
                shutil.copy2(script, scripts / script.name)
            out = root / 'outputs with spaces 测试'
            env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONIOENCODING='utf-8')
            result = subprocess.run(self.command(scripts / 'sparkle_multibackend.py', out),
                                    env=env, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = json.loads((out / 'run.json').read_text(encoding='utf-8'))
            stats = json.loads((out / 'chromatix/stats.json').read_text(encoding='utf-8'))
            context = stats['execution_context']
            self.assertEqual(run['status'], 'complete')
            self.assertEqual(run['environment']['environment_settings']['JAX_PLATFORMS'], 'cpu')
            self.assertEqual(context['runtime'], 'wsl')
            self.assertEqual(context['default_backend'], 'gpu')
            self.assertTrue(all('cuda' in device for device in context['devices']))
            self.assertTrue(context['environment']['executable'].startswith('/'))
            self.assertEqual(context['environment']['environment_settings']['JAX_PLATFORMS'], 'cuda')
            self.assertIsNotNone(context['environment']['versions']['jax-cuda13-plugin'])
            self.assertEqual(stats['num_sources_or_rays'], 16)

    def test_failed_wsl_worker_is_reported_without_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            env = dict(os.environ, SPARKLE_WSL_PYTHON='/missing/sparkle-python',
                       JAX_PLATFORMS='cpu', PYTHONIOENCODING='utf-8')
            result = subprocess.run(self.command(ROOT / 'scripts/sparkle_multibackend.py', out),
                                    env=env, capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(result.returncode, 0)
            run = json.loads((out / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual(run['status'], 'failed')
            self.assertEqual(run['backend_runs']['chromatix']['status'], 'failed')
            self.assertIn('WSL worker failed', run['backend_runs']['chromatix']['error']['message'])
            self.assertFalse((out / 'chromatix/stats.json').exists())
            self.assertTrue((out / 'report.html').is_file())


if __name__ == '__main__':
    unittest.main()
