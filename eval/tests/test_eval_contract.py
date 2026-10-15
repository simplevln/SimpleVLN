"""Dependency-free checks for endpoint configuration and evaluation orchestration."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'vln/eval.py').read_text())


class EvalContractTests(unittest.TestCase):
    def resolve(self, **kwargs):
        args = SimpleNamespace(vllm_base_url=None, vllm_host='localhost',
                               vllm_port_start=22001, vllm_num_servers=2)
        vars(args).update(kwargs)
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'resolve_vllm_urls')
        ns = {'urlsplit': urlsplit}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<urls>', 'exec'), ns)
        return ns['resolve_vllm_urls'](args)

    def test_server_urls(self):
        self.assertEqual(self.resolve(), 'http://localhost:22001/v1,http://localhost:22002/v1')
        self.assertEqual(self.resolve(vllm_host='https://example.com', vllm_num_servers=1),
                         'https://example.com:22001/v1')
        self.assertEqual(self.resolve(vllm_base_url='https://a:8001/v1, http://b:8002/v1/'),
                         'https://a:8001/v1,http://b:8002/v1')
        self.assertEqual(self.resolve(vllm_host='[::1]', vllm_num_servers=1),
                         'http://[::1]:22001/v1')
        for bad in ({'vllm_base_url': ' , '}, {'vllm_num_servers': 0},
                    {'vllm_port_start': 65535}, {'vllm_host': 'localhost:8001'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.resolve(**bad)

    def test_incomplete_run_cannot_produce_summary(self):
        main = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'run_evaluation')
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'failed_workers' for t in n.targets))
        guard = compile(ast.Module(body=main.body[start:start + 2], type_ignores=[]), '<guard>', 'exec')
        for exitcode, count in ((1, 2), (0, 1)):
            with self.subTest(exitcode=exitcode, count=count), self.assertRaises(RuntimeError):
                exec(guard, {'processes': [SimpleNamespace(pid=1, exitcode=exitcode)],
                             'done_count': count, 'target': 2})
        exec(guard, {'processes': [SimpleNamespace(pid=1, exitcode=0)],
                     'done_count': 2, 'target': 2})


if __name__ == '__main__':
    unittest.main()
