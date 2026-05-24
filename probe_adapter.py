"""Direct inspection: does the safetensors file PEFT *would* load contain
the trained B weight, or is B literally zeros on disk?

If on-disk B is non-zero but the loaded model's B is zero, PEFT is silently
failing to bind the weights. We also wrap PEFT's load in a warnings.catch_warnings
to surface any UserWarning peft normally emits about missing/skipped keys.
"""
import os, sys, warnings, json, socket
sys.path.insert(0, "/home/csgrad/mbhosale/phd/SCALE/MAGMAR-MWV")

print(f"=== HOST: {socket.gethostname()} ===", flush=True)

from huggingface_hub import snapshot_download
sp = snapshot_download(repo_id="AdoptedIrelia/UNLI", allow_patterns=["lora/*"])
lora_dir = os.path.join(sp, "lora")
print(f"snapshot path: {sp}", flush=True)
print(f"lora dir:      {lora_dir}", flush=True)
print(f"lora dir contents:", flush=True)
for f in sorted(os.listdir(lora_dir)):
    full = os.path.join(lora_dir, f)
    sz = os.path.getsize(full) if os.path.isfile(full) else "(dir)"
    print(f"  {f:50s} {sz}", flush=True)

# 1) Inspect adapter_model.safetensors on disk
from safetensors import safe_open
import torch

st_path = os.path.join(lora_dir, "adapter_model.safetensors")
print(f"\n=== on-disk safetensors {st_path} ===", flush=True)
with safe_open(st_path, framework="pt") as f:
    keys = list(f.keys())
    print(f"  total keys: {len(keys)}", flush=True)
    # Find lm_head lora_A and lora_B
    a_keys = [k for k in keys if "lm_head" in k and "lora_A" in k]
    b_keys = [k for k in keys if "lm_head" in k and "lora_B" in k]
    print(f"  lm_head lora_A keys: {a_keys}", flush=True)
    print(f"  lm_head lora_B keys: {b_keys}", flush=True)
    for k in (a_keys + b_keys):
        t = f.get_tensor(k)
        print(f"    {k}: shape={tuple(t.shape)} dtype={t.dtype} norm={t.float().norm().item():.6f}  max_abs={t.float().abs().max().item():.6f}", flush=True)

# 2) Inspect adapter_config to find what target_modules expect
cfg_path = os.path.join(lora_dir, "adapter_config.json")
cfg = json.load(open(cfg_path))
print(f"\n=== adapter_config.json ===", flush=True)
print(f"  base_model_name_or_path: {cfg.get('base_model_name_or_path')}", flush=True)
print(f"  peft_type:               {cfg.get('peft_type')}", flush=True)
print(f"  r, lora_alpha, scaling:  r={cfg.get('r')}, alpha={cfg.get('lora_alpha')}", flush=True)
print(f"  target_modules contains lm_head: {'lm_head' in (cfg.get('target_modules') or [])}", flush=True)
print(f"  modules_to_save:         {cfg.get('modules_to_save')}", flush=True)

# 3) Capture warnings while peft loads
print(f"\n=== PEFT load with full warning capture ===", flush=True)
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from peft import PeftModel

base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "AdoptedIrelia/UNLI",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
)
print(f"base model device: {next(base.parameters()).device}", flush=True)

# Untie like the patch does
inner = getattr(base, "thinker", base)
lm_head = inner.lm_head
print(f"BEFORE untie, lm_head.weight norm = {lm_head.weight.float().norm().item():.4f}", flush=True)
lm_head.weight = torch.nn.Parameter(lm_head.weight.detach().clone())
inner.config.tie_word_embeddings = False
base.config.tie_word_embeddings = False
base.tie_weights = lambda: None

with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    peft_model = PeftModel.from_pretrained(base, lora_dir)
    print(f"  peft.from_pretrained captured {len(wlist)} warnings:", flush=True)
    for w in wlist:
        print(f"    [{w.category.__name__}] {w.message}", flush=True)

# Inspect loaded B norm
inner2 = getattr(getattr(peft_model, "base_model", peft_model), "model", peft_model)
inner2 = getattr(inner2, "thinker", inner2)
lm_after = inner2.lm_head
print(f"\nloaded model lm_head class: {type(lm_after).__name__}", flush=True)
if hasattr(lm_after, "lora_B"):
    for name, mod in lm_after.lora_B.items():
        b = mod.weight
        a = lm_after.lora_A[name].weight
        print(f"  loaded adapter={name}: A norm={a.float().norm().item():.4f}  B norm={b.float().norm().item():.6f}", flush=True)
