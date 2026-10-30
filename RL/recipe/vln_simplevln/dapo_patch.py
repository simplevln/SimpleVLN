"""DAPO isolation switch — swap in VLNDAPORayTrainer, opt-in via VERL_USE_EXTERNAL_MODULES.

Load order: ``verl/__init__.py`` calls ``import_external_libs`` on every comma-separated
entry of ``VERL_USE_EXTERNAL_MODULES`` at verl-package import time — which is BEFORE
``verl.trainer.main_ppo``'s ``TaskRunner.run()`` instantiates the trainer. Importing this
module therefore:

  1. imports ``recipe.vln_simplevln.patches`` first, so the VLN runtime patches (mm collation
     fix, progress, trajectory-level GRPO advantage, padding-free data metrics) are installed
     for the DAPO run exactly as they are for GRPO;
  2. installs VLNDAPORayTrainer's Dynamic Sampling onto the trainer the entrypoint
     actually instantiates.

WHY rebinding the module global is NOT enough (the subtle bug this file fixes):
the entrypoint is ``python -m verl.trainer.main_ppo``, so the file runs as ``__main__``
and ``TaskRunner`` is ``__main__.TaskRunner`` — a DIFFERENT module object than
``verl.trainer.main_ppo`` (which -m also imports as a side effect). ``run_ppo`` ships
``__main__.TaskRunner`` to a Ray actor; because ``__main__`` isn't importable remotely,
Ray cloudpickles it BY VALUE and captures ``run()``'s ``RayPPOTrainer`` global from the
driver's ``__main__`` namespace — the STOCK class. So ``main_ppo.py:299`` instantiates
the stock trainer and rebinding ``verl.trainer.main_ppo.RayPPOTrainer`` never reaches it
(observed: the swap log prints but stock ``fit()`` runs → no Dynamic Sampling).

The fix: monkeypatch the stock ``RayPPOTrainer`` CLASS OBJECT in place. That object is
shared by reference across ``__main__``, ``verl.trainer.main_ppo`` and the actor's
re-import (it lives in the importable ``verl.trainer.ppo.ray_trainer``, so cloudpickle
pickles it by reference), so replacing its methods reaches whatever instance line 299
creates, on any node.

The run scripts opt into this patch by setting
``VERL_USE_EXTERNAL_MODULES=recipe.vln_simplevln.patches,recipe.vln_simplevln.dapo_patch``.
"""

# (1) install the shared VLN runtime patches (idempotent import).
from recipe.vln_simplevln import patches as _vln_patches
_ = _vln_patches  # keep import: patches module has import-time side effects

# (2) install DAPO Dynamic Sampling onto the trainer the entrypoint instantiates.
import verl.trainer.main_ppo as _main_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _StockTrainer

from recipe.vln_simplevln.vln_dapo_trainer import VLNDAPORayTrainer

# (2a) rebind the module global too — harmless belt-and-suspenders for any path that
# resolves RayPPOTrainer via the verl.trainer.main_ppo module dict.
_main_ppo.RayPPOTrainer = VLNDAPORayTrainer

# (2b) the load-bearing part: copy VLNDAPORayTrainer's own methods (fit + DAPO helpers)
# onto the stock class object in place. Uses vars() to preserve descriptor types — in
# particular _valid_uids is a staticmethod and must stay one (it's called as
# self._valid_uids(...)). Skips dunders so we only move the DAPO additions/overrides.
_patched = []
for _name, _attr in vars(VLNDAPORayTrainer).items():
    if _name.startswith("__"):
        continue
    setattr(_StockTrainer, _name, _attr)
    _patched.append(_name)

print(
    f"[vln-dapo-patch] RayPPOTrainer <- VLNDAPORayTrainer methods {sorted(_patched)} "
    f"(module global also rebound)"
)
