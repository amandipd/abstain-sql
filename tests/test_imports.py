"""Every CPU-side module imports from a clean checkout.

GPU-side modules (train/grpo_train.py, eval/generate.py) are left out: they will
import TRL and vLLM, which the local environment does not install.
"""

import importlib

import pytest

CPU_MODULES = [
    "data.generator",
    "data.generator.schemas",
    "checker.execute",
    "checker.testsuite",
    "train.reward",
    "calib.signals",
    "calib.ltt",
    "eval.metrics",
    "eval.baselines",
    "eval.bootstrap",
    "eval.local_gguf",
]


@pytest.mark.parametrize("name", CPU_MODULES)
def test_module_imports(name):
    importlib.import_module(name)
