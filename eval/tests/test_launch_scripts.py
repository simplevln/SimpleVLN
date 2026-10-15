import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def test_server_health_and_gpu_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            fake = path / 'vllm'
            fake.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
Path(os.environ['CAPTURE']).write_text(json.dumps({'devices': os.environ['CUDA_VISIBLE_DEVICES'], 'args': sys.argv[1:]}))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/health' else 404)
        self.end_headers()
    def log_message(self, *args): pass
HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port') + 1])), Handler).serve_forever()
''')
            fake.chmod(0o755)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = {**os.environ, 'PATH': str(path) + os.pathsep + os.environ['PATH'],
                   'MODEL_PATH': 'mock', 'LOG_DIR': str(path / 'logs'), 'BASE_PORT': str(port),
                   'CUDA_VISIBLE_DEVICES': '4,7', 'NUM_GPUS': '2', 'TP_SIZE': '2',
                   'STARTUP_TIMEOUT': '5', 'PYTHON': sys.executable, 'CAPTURE': str(path / 'capture'),
                   'http_proxy': 'http://127.0.0.1:1', 'HTTP_PROXY': 'http://127.0.0.1:1',
                   'no_proxy': '', 'NO_PROXY': ''}
            try:
                result = subprocess.run(['bash', str(ROOT / 'scripts/serve_vllm.sh')],
                                        env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                captured = json.loads((path / 'capture').read_text())
                self.assertEqual(captured['devices'], '4,7')
                self.assertNotIn('--compilation-config', captured['args'])
                self.assertIn('is ready', result.stdout)
            finally:
                for pid_file in (path / 'logs').glob('*/server.pids'):
                    for pid in pid_file.read_text().splitlines():
                        try: os.kill(int(pid), signal.SIGTERM)
                        except ProcessLookupError: pass

    def test_occupied_port_and_invalid_tp_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as folder, socket.socket() as sock:
            sock.bind(('0.0.0.0', 0))
            env = {**os.environ, 'MODEL_PATH': 'mock', 'LOG_DIR': folder,
                   'BASE_PORT': str(sock.getsockname()[1]), 'CUDA_VISIBLE_DEVICES': '0',
                   'PYTHON': sys.executable}
            for tp in ('0', '1'):
                result = subprocess.run(['bash', str(ROOT / 'scripts/serve_vllm.sh'), '1', tp],
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(list(Path(folder).glob('*/server.pids')))

    def test_disabled_or_empty_gpu_selection_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as folder:
            env = {**os.environ, 'MODEL_PATH': 'mock', 'LOG_DIR': folder,
                   'PYTHON': sys.executable}
            for devices in ('', '-1', '0,', ',0', '0,,1'):
                with self.subTest(devices=devices):
                    result = subprocess.run(
                        ['bash', str(ROOT / 'scripts/serve_vllm.sh'), '1', '1'],
                        env={**env, 'CUDA_VISIBLE_DEVICES': devices},
                        capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('CUDA_VISIBLE_DEVICES', result.stdout)
                    self.assertFalse(list(Path(folder).glob('*/server.pids')))

    def test_eval_forwards_explicit_urls_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            fake = path / 'python with spaces'
            fake.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
with Path(os.environ['CAPTURE']).open('a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
sys.exit(9)
''')
            fake.chmod(0o755)
            env = {**os.environ, 'PYTHON': str(fake), 'EXP_NAME': 'mock',
                   'CAPTURE': str(path / 'capture'), 'VLLM_BASE_URL': 'https://host:8443/v1'}
            result = subprocess.run(['bash', str(ROOT / 'scripts/eval.sh')],
                                    cwd=folder, env=env, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 9)
            lines = (path / 'capture').read_text().splitlines()
            self.assertEqual(len(lines), 1)
            argv = json.loads(lines[0])
            self.assertEqual(argv[argv.index('--vllm_base_url') + 1], env['VLLM_BASE_URL'])


if __name__ == '__main__':
    unittest.main()
