#!/usr/bin/env bash
#
# LazyAttention installer.
#
#   bash scripts/install.sh                 # default: prebuilt vLLM wheel + lazy_attn (~2 min)
#   bash scripts/install.sh --venv .venv    # ... into a fresh virtualenv
#   bash scripts/install.sh --bench         # ... plus the benchmark dependencies
#   bash scripts/install.sh --source        # build vLLM from source instead (see vllm_proj/)
#   bash scripts/install.sh --check         # only report on the environment, install nothing
#
# LazyAttention and BlockAttention are pure-Python monkey patches over vLLM: the
# hot path is Triton (JIT-compiled at runtime), so the stock vLLM wheel is all
# the compiled code you need. Build from source only if you are changing vLLM's
# own C++/CUDA.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VLLM_VERSION="0.8.5.post1"
EXPECTED_TORCH="2.6.0"

FROM_SOURCE=0
WITH_BENCH=0
CHECK_ONLY=0
VERIFY=1
VENV_PATH=""

# ---------------------------------------------------------------- args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     FROM_SOURCE=1; shift ;;
        --bench)      WITH_BENCH=1; shift ;;
        --check)      CHECK_ONLY=1; shift ;;
        --no-verify)  VERIFY=0; shift ;;
        --venv)       VENV_PATH="${2:?--venv needs a path}"; shift 2 ;;
        -h|--help)    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *)            echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# -------------------------------------------------------------- venv -----
if [[ -n "${VENV_PATH}" ]]; then
    say "Creating virtualenv at ${VENV_PATH}"
    python3 -m venv "${VENV_PATH}"
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
    pip install --quiet --upgrade pip
fi

PY="$(command -v python3 || true)"
[[ -n "${PY}" ]] || die "python3 not found on PATH"

# ------------------------------------------------------------- checks ----
say "Environment"
"${PY}" - <<'EOF'
import platform, shutil, subprocess, sys

print(f"  python   {platform.python_version()}  ({sys.executable})")
if not (3, 9) <= sys.version_info[:2] <= (3, 12):
    print("  ^ vLLM 0.8.5.post1 supports Python 3.9-3.12", file=sys.stderr)

nvidia_smi = shutil.which("nvidia-smi")
if nvidia_smi:
    out = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.total,compute_cap",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    for line in out.splitlines():
        print(f"  gpu      {line}")
else:
    print("  gpu      no nvidia-smi found")

nvcc = shutil.which("nvcc")
print(f"  nvcc     {nvcc or 'not found (only needed for --source)'}")
EOF

if [[ ${CHECK_ONLY} -eq 1 ]]; then
    say "Check only; nothing installed."
    exit 0
fi

# -------------------------------------------------------------- vllm ----
if [[ ${FROM_SOURCE} -eq 1 ]]; then
    say "Building vLLM ${VLLM_VERSION} from source (this takes 20-60 min)"
    bash "${REPO_ROOT}/vllm_proj/install.sh"
else
    say "Installing vLLM ${VLLM_VERSION} (prebuilt wheel)"
    pip install "vllm==${VLLM_VERSION}"
fi

# ---------------------------------------------------------- lazy_attn ----
say "Installing lazy_attn (editable)"
pip install -e "${REPO_ROOT}/lazy_attn"

if [[ ${WITH_BENCH} -eq 1 ]]; then
    say "Installing benchmark dependencies"
    pip install -r "${REPO_ROOT}/benchmarks/requirements.txt"
fi

# ------------------------------------------------------------ verify ----
if [[ ${VERIFY} -eq 0 ]]; then
    say "Done (verification skipped)."
    exit 0
fi

say "Verifying"
"${PY}" - "${EXPECTED_TORCH}" <<'EOF'
import sys

expected_torch = sys.argv[1]
problems = []

import torch
print(f"  torch    {torch.__version__} (cuda {torch.version.cuda})")
if not torch.__version__.startswith(expected_torch):
    problems.append(
        f"torch is {torch.__version__}, but vLLM 0.8.5.post1 is built against "
        f"{expected_torch}; its compiled kernels will not load."
    )

import vllm
print(f"  vllm     {vllm.__version__}")

try:
    import vllm._C  # noqa: F401
    print("  vllm._C  loaded")
except ImportError as exc:
    problems.append(
        f"vllm._C failed to import ({exc}). The vLLM install has no compiled "
        "kernels -- a source build that silently failed leaves it in this state."
    )

import lazy.__vllm__  # noqa: F401  (applies the LazyAttention patches)
print("  lazy     patches applied")

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    name = torch.cuda.get_device_name(0)
    supported = torch.cuda.get_arch_list()
    print(f"  device   {name} ({arch})")
    if arch not in supported:
        problems.append(
            f"{name} is {arch}, but this torch only ships {', '.join(supported)}. "
            "Every CUDA kernel will fail with 'no kernel image is available'. "
            "See scripts/install_sm120.sh for Blackwell (sm_120) consumer cards."
        )
else:
    problems.append("torch.cuda.is_available() is False -- no usable GPU.")

if problems:
    print("", file=sys.stderr)
    for p in problems:
        print(f"\033[31mFAIL:\033[0m {p}", file=sys.stderr)
    sys.exit(1)
EOF

say "Done. Try: bash scripts/validate.sh"
