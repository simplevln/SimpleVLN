"""Launch the SimpleVLN habitat env_server.

Takes ONE unified habitat config that lists every dataset under a top-level
``datasets:`` map. Each parquet row's ``extra_info.dataset_name`` picks which
dataset the requested episode belongs to at reset().

Example:

  python -m recipe.vln_simplevln.env_server.launch \\
      --config recipe/vln_simplevln/config/vln_datasets.yaml \\
      --port 8002 --pool-size 8 --gpu-ids 0,1,2,3,4,5,6,7

Each worker owns one GPU (CUDA_VISIBLE_DEVICES); if ``--gpu-ids`` has fewer
entries than ``--pool-size`` they cycle round-robin.
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import uvicorn  # noqa: E402

from recipe.vln_simplevln.env_server.server import app, set_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--config", required=True,
                   help="Unified habitat config yaml with a `datasets:` map at top level.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--pool-size", type=int, default=1,
                   help="Number of parallel habitat workers (one process per GPU).")
    p.add_argument("--session-ttl-sec", type=float, default=1800.0)
    p.add_argument("--gpu-ids", type=str, default=None,
                   help="Comma-separated GPU IDs, one per worker (e.g. 0,1,2,3,4,5,6,7).")
    args = p.parse_args()

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else None

    set_config(args.config, pool_size=args.pool_size,
               session_ttl_sec=args.session_ttl_sec, gpu_ids=gpu_ids)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
