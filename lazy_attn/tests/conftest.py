# lazyattn/tests/conftest.py
import importlib
import os
import sys
import pytest

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECK_MODULES = True

# ================================================
# os.environ['PYTORCH_CUDA_GRAPH_DEBUG'] = '1'
# ================================================

# Add lazyattn directory to python path
lazyattn_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lazy_attn"))
sys.path.insert(0, lazyattn_root)
logger.info(f"Added lazyattn to Python path: {lazyattn_root}")

# Check the availability of vllm and lazy modules
if CHECK_MODULES:
    try:
        import vllm
        import lazy
    except ImportError as e:
        logger.warning(f"Failed to import module: {e}")

# These modules import helpers from vLLM's own `tests.*` package, which only
# exists in a vLLM *source* checkout -- it is not shipped in the wheel that
# `scripts/install.sh` installs. Collecting them there raises ImportError,
# which aborts the entire run before a single supported test executes. Skip
# them when the helpers are absent, and collect them when they are.
SOURCE_ONLY_TESTS = frozenset({
    "kernels/test_prefix_prefill.py",
    "kernels/test_rope.py",
    "core/test_kv_cache_utils.py",
})

# Probes covering both halves of the tree those modules pull from.
_VLLM_TEST_HELPERS = ("tests.kernels.allclose_default",
                      "tests.v1.core.test_kv_cache_utils")

_TESTS_ROOT = os.path.dirname(os.path.abspath(__file__))
_vllm_source_tests: "bool | None" = None


def _has_vllm_source_tests() -> bool:
    global _vllm_source_tests
    if _vllm_source_tests is None:
        _vllm_source_tests = True
        for module in _VLLM_TEST_HELPERS:
            try:
                importlib.import_module(module)
            except Exception:
                _vllm_source_tests = False
                break
    return _vllm_source_tests


def pytest_ignore_collect(collection_path, config):
    try:
        rel = os.path.relpath(str(collection_path), _TESTS_ROOT)
    except ValueError:
        return None
    if rel.replace(os.sep, "/") not in SOURCE_ONLY_TESTS:
        return None
    if _has_vllm_source_tests():
        return None
    logger.warning(
        "Skipping %s: it imports vLLM's own tests.* helpers, which the "
        "prebuilt wheel does not ship. Run against a vLLM source checkout "
        "to include it.", rel)
    return True


@pytest.fixture(scope="session")
def mock_prompts():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    return prompts


@pytest.fixture(scope="session")
def mock_sampling_params():
    from vllm import SamplingParams
    return SamplingParams(temperature=0, max_tokens=20, min_tokens=20, seed=42)


# Default to the 1B block-fine-tuned model so the suite fits on a single
# consumer GPU. Point LAZY_TEST_MODEL at ldsjmdy/Tulu3-Block-FT (8B) to
# reproduce the paper's accuracy numbers.
DEFAULT_TEST_MODEL = "hxia7/Llama-3.2-1B-Block-FT"


@pytest.fixture(scope="session")
def mock_model_name():
    return os.environ.get("LAZY_TEST_MODEL", DEFAULT_TEST_MODEL)


@pytest.fixture(scope="session")
def mock_gpu_memory_utilization():
    return float(os.environ.get("LAZY_TEST_GPU_MEM_UTIL", "0.6"))
