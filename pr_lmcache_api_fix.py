from pathlib import Path


files = [
    Path("/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_v1_adapter.py"),
    Path("/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_adapter.py"),
]

for p in files:
    if not p.exists():
        print("skip missing", p)
        continue
    s = p.read_text()
    s2 = s.replace(
        "from vllm.utils import get_kv_cache_torch_dtype\nfrom vllm.utils.math_utils import cdiv",
        "from vllm.utils.math_utils import cdiv\nfrom vllm.utils.torch_utils import get_kv_cache_torch_dtype",
    )
    s2 = s2.replace(
        "from vllm.utils import cdiv, get_kv_cache_torch_dtype",
        "from vllm.utils.math_utils import cdiv\nfrom vllm.utils.torch_utils import get_kv_cache_torch_dtype",
    )
    s2 = s2.replace(
        "from vllm.utils import cdiv, round_down",
        "from vllm.utils.math_utils import cdiv, round_down",
    )
    if s2 != s:
        p.write_text(s2)
        print("patched", p)
    else:
        print("no change", p)
