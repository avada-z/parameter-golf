"""Starter training script; keep under 1500 lines."""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    val_log_every = int(os.environ.get("VAL_LOG_EVERY", 10))
    validate_at_step_zero = bool(int(os.environ.get("VALIDATE_AT_STEP_ZERO", "0")))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    linear_impl = os.environ.get("LINEAR_IMPL", "standard")
    embed_impl = os.environ.get("EMBED_IMPL", "dense")
    head_impl = os.environ.get("HEAD_IMPL", "dense")
    sine_k = int(os.environ.get("SINE_K", 16))
    sine_m = int(os.environ.get("SINE_M", 1))
    attention_mode = os.environ.get("ATTENTION_MODE", "binary_relu2")
    binary_optimizer = os.environ.get("BINARY_OPTIMIZER", "muon")
    bop_adaptivity = float(os.environ.get("BOP_ADAPTIVITY", 1e-3))
    bop_threshold = float(os.environ.get("BOP_THRESHOLD", 1e-4))
    size_gated_rank = int(os.environ.get("SIZE_GATED_RANK", 0))
    size_loss_weight = float(os.environ.get("SIZE_LOSS_WEIGHT", 0.0))
    size_gate_init = float(os.environ.get("SIZE_GATE_INIT", -6.0))
    size_bytes_per_param = float(os.environ.get("SIZE_BYTES_PER_PARAM", 1.0))
    size_export_threshold = float(os.environ.get("SIZE_EXPORT_THRESHOLD", 0.5))
    attn_chunk_size = int(os.environ.get("ATTN_CHUNK_SIZE", 128))
    checkpoint_blocks = bool(int(os.environ.get("CHECKPOINT_BLOCKS", "0")))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    binary_embed_width_mult = int(os.environ.get("BINARY_EMBED_WIDTH_MULT", 1))
    binary_embed_layers = int(os.environ.get("BINARY_EMBED_LAYERS", 1))
    binary_lm_head_norm = bool(int(os.environ.get("BINARY_LM_HEAD_NORM", "1")))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    use_compile = bool(int(os.environ.get("USE_COMPILE", "1")))
    compile_fullgraph = bool(int(os.environ.get("COMPILE_FULLGRAPH", "0")))
    compile_disable_mix_order_reduction = bool(int(os.environ.get("COMPILE_DISABLE_MIX_ORDER_REDUCTION", "1")))
    compile_disable_epilogue_fusion = bool(int(os.environ.get("COMPILE_DISABLE_EPILOGUE_FUSION", "0")))

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def configure_inductor_for_binary_compile(args: Hyperparameters) -> list[str]:
    tweaks: list[str] = []
    if not args.use_compile:
        return tweaks
    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return tweaks
    if args.compile_disable_mix_order_reduction and hasattr(inductor_config, "triton"):
        inductor_config.triton.mix_order_reduction = False
        tweaks.append("no_mix_order_reduction")
    if args.compile_disable_epilogue_fusion and hasattr(inductor_config, "epilogue_fusion"):
        inductor_config.epilogue_fusion = False
        tweaks.append("no_epilogue_fusion")
    return tweaks


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


class BOP(torch.optim.Optimizer):
    def __init__(self, params, threshold_map: dict[int, Tensor], adaptivity: float, threshold: float):
        super().__init__(params, dict(lr=1.0, adaptivity=adaptivity, threshold=threshold))
        self.threshold_map = threshold_map

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            a = group["adaptivity"] * group["lr"]
            tflip = group["threshold"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                m = self.state[p].setdefault("m", torch.zeros_like(p, dtype=torch.float32))
                m.mul_(1 - a).add_(p.grad.float(), alpha=a)
                thr = self.threshold_map[id(p)].float()
                centered = p.float() - thr
                mask = (m.abs() >= tflip) & (m.sign() == centered.sign())
                centered[mask].neg_()
                p.copy_((centered + thr).to(dtype=p.dtype))
                m[mask] = 0
        return loss


class SizeGatedAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, gate_init: float, bytes_per_param: float):
        super().__init__()
        self.in_features, self.out_features, self.bytes_per_param = in_features, out_features, bytes_per_param
        self.down = nn.Parameter(torch.randn(rank, in_features) * (1.0 / max(1, in_features) ** 0.5))
        self.up = nn.Parameter(torch.zeros(out_features, rank))
        self.gate_logits = nn.Parameter(torch.full((rank,), gate_init))

    def gate_probs(self) -> Tensor: return torch.sigmoid(self.gate_logits)
    def forward(self, x: Tensor) -> Tensor:
        z = self.gate_probs().to(dtype=x.dtype)
        return F.linear(F.linear(x, self.down.to(dtype=x.dtype)) * z, self.up.to(dtype=x.dtype))
    def expected_size_bytes(self) -> Tensor:
        return self.gate_probs().sum() * (self.in_features + self.out_features) * self.bytes_per_param


def module_expected_size_bytes(module: nn.Module) -> Tensor:
    param = next(iter(module.parameters()), None)
    total = None
    for submodule in module.modules():
        adapter = getattr(submodule, "size_adapter", None)
        if adapter is not None:
            val = adapter.expected_size_bytes()
            total = val if total is None else total + val
    return total if total is not None else torch.zeros((), device=param.device if param is not None else "cpu")


def export_state_dict_with_size_gates(module: nn.Module, threshold: float) -> dict[str, Tensor]:
    state = module.state_dict()
    for name, submodule in module.named_modules():
        adapter = getattr(submodule, "size_adapter", None)
        if adapter is None:
            continue
        prefix = f"{name}.size_adapter." if name else "size_adapter."
        keep = (adapter.gate_probs() >= threshold).nonzero(as_tuple=False).flatten()
        state.pop(prefix + "down", None); state.pop(prefix + "up", None)
        state[prefix + "active_idx"] = keep.to(dtype=torch.int32)
        state[prefix + "active_down"] = adapter.down.detach().index_select(0, keep)
        state[prefix + "active_up"] = adapter.up.detach().index_select(1, keep)
    return state


def load_state_dict_with_size_gates(module: nn.Module, state_dict: dict[str, Tensor], strict: bool = True) -> None:
    expanded = dict(state_dict)
    for name, submodule in module.named_modules():
        adapter = getattr(submodule, "size_adapter", None)
        prefix = f"{name}.size_adapter." if name else "size_adapter."
        if adapter is None or prefix + "active_idx" not in expanded:
            continue
        idx = expanded.pop(prefix + "active_idx").to(dtype=torch.long)
        active_down = expanded.pop(prefix + "active_down")
        active_up = expanded.pop(prefix + "active_up")
        down, up = torch.zeros_like(adapter.down), torch.zeros_like(adapter.up)
        if idx.numel() > 0:
            down.index_copy_(0, idx, active_down.to(dtype=down.dtype))
            up.index_copy_(1, idx, active_up.to(dtype=up.dtype))
        expanded[prefix + "down"], expanded[prefix + "up"] = down, up
    module.load_state_dict(expanded, strict=strict)

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    log_fn=None,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    total_batches = max((seq_end - seq_start + local_batch_seqs - 1) // local_batch_seqs, 1)
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        eval_t0 = time.perf_counter()
        for batch_idx, batch_seq_start in enumerate(range(seq_start, seq_end, local_batch_seqs), start=1):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
            if (
                log_fn is not None
                and rank == 0
                and args.val_log_every > 0
                and (batch_idx == 1 or batch_idx % args.val_log_every == 0 or batch_idx == total_batches)
            ):
                elapsed_ms = 1000.0 * (time.perf_counter() - eval_t0)
                log_fn(
                    f"validating_progress:batches:{batch_idx}/{total_batches} "
                    f"elapsed:{elapsed_ms:.0f}ms"
                )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def find_binary_weight_bases(state_dict: dict[str, Tensor]) -> dict[str, str]:
    bases: dict[str, str] = {}
    for name, tensor in state_dict.items():
        if not name.endswith(".threshold"):
            continue
        base = name.removesuffix(".threshold")
        weight_name = f"{base}.weight"
        if weight_name in state_dict and state_dict[weight_name].ndim == 2:
            bases[weight_name] = name
    return bases


def pack_binary_tensor(sign_tensor: Tensor) -> Tensor:
    bits = sign_tensor.to(dtype=torch.uint8).cpu().numpy()
    packed = np.packbits(bits, axis=-1, bitorder="little")
    return torch.from_numpy(packed.copy())


def unpack_binary_tensor(packed: Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    bits = np.unpackbits(packed.cpu().numpy(), axis=-1, bitorder="little")
    bits = bits[..., : shape[-1]]
    signs = bits.astype(np.float32) * 2.0 - 1.0
    return torch.from_numpy(signs.reshape(shape)).to(dtype=dtype).contiguous()

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    binary: dict[str, Tensor] = {}
    binary_meta: dict[str, dict[str, object]] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    binary_thresholds = find_binary_weight_bases(state_dict)
    consumed_names: set[str] = set()

    for name, tensor in state_dict.items():
        if name in consumed_names:
            continue
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        threshold_name = binary_thresholds.get(name)
        if threshold_name is not None:
            threshold = state_dict[threshold_name].detach().to("cpu").contiguous()
            stats["param_count"] += int(threshold.numel())
            stats["num_tensors"] += 1
            stats["baseline_tensor_bytes"] += tensor_nbytes(threshold)
            sign_tensor = (t.float() - threshold.float()).ge(0)
            packed = pack_binary_tensor(sign_tensor)
            binary[name] = packed
            binary_meta[name] = {
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "threshold_name": threshold_name,
                "threshold_shape": list(threshold.shape),
                "threshold_dtype": str(threshold.dtype).removeprefix("torch."),
            }
            stats["num_float_tensors"] += 1
            stats["int8_payload_bytes"] += tensor_nbytes(packed)
            consumed_names.add(threshold_name)
            continue

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "mixed_binary_int8_clean_v1",
        "binary": binary,
        "binary_meta": binary_meta,
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, packed in obj.get("binary", {}).items():
        meta = obj["binary_meta"][name]
        dtype = getattr(torch, meta["dtype"])
        shape = tuple(int(v) for v in meta["shape"])
        out[name] = unpack_binary_tensor(packed, shape, dtype)
        threshold_name = meta["threshold_name"]
        threshold_dtype = getattr(torch, meta["threshold_dtype"])
        threshold_shape = tuple(int(v) for v in meta["threshold_shape"])
        out[threshold_name] = torch.zeros(threshold_shape, dtype=threshold_dtype).contiguous()
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def binarize(x: Tensor) -> Tensor:
    hard = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    surrogate = torch.tanh(x)
    return hard.detach() + surrogate - surrogate.detach()


def round_ste(x: Tensor) -> Tensor:
    rounded = x.round()
    return x + (rounded - x).detach()
def strict_sign(x: Tensor) -> Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
def scale_last_dim(x: Tensor, scale: Tensor) -> Tensor:
    x_flat = x.reshape(-1, x.size(-1))
    y_flat = x_flat * scale.to(dtype=x.dtype)
    return y_flat.view_as(x)
def scale_head_axis(x: Tensor, scale: Tensor) -> Tensor:
    bsz, nheads, seqlen, head_dim = x.shape
    x_perm = x.permute(0, 2, 3, 1).reshape(-1, nheads)
    y_perm = x_perm * scale.to(dtype=x.dtype)
    return y_perm.view(bsz, seqlen, head_dim, nheads).permute(0, 3, 1, 2).contiguous()


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


def binary_shift_scale(shift_param: Tensor, lo: int = -8, hi: int = 0) -> Tensor:
    return 2 ** round_ste(shift_param.clamp(lo, hi))
def binary_linear_weight(weight: Tensor, threshold: Tensor) -> Tensor:
    return binarize(weight - threshold)


def maybe_add_size_adapter(module: nn.Module, x: Tensor, out: Tensor) -> Tensor:
    adapter = getattr(module, "size_adapter", None)
    return out if adapter is None else out + adapter(x)


def make_sine_bank(m: int, k: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 31 * m + k)
    freq = (torch.rand(m, k, generator=g) * (2 * math.pi * math.log2(max(2, k))) + 1.0) / math.sqrt(max(1, m))
    phase = torch.rand(k, generator=g) * (2 * math.pi)
    r = min(k, 8)
    sketch = (torch.randint(0, 2, (k, r), generator=g).float() * 2 - 1) / math.sqrt(max(1, k))
    return freq, phase, sketch


class SineLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, latent: Tensor, freq: Tensor, phase: Tensor, sketch: Tensor, m: int, k: int) -> Tensor:
        lat = latent.view(latent.shape[0], latent.shape[1] // m, m)
        ang = torch.einsum("obm,mk->obk", lat, freq) + phase.view(1, 1, -1)
        w_full = strict_sign(torch.sin(ang)).reshape(latent.shape[0], -1)
        x_flat = x.reshape(-1, lat.shape[1], k)
        ctx.save_for_backward(torch.einsum("nik,kr->nir", x_flat, sketch.to(dtype=x.dtype)), latent, freq, phase, sketch)
        ctx.m = m
        ctx.k = k
        ctx.xshape = x.shape
        return F.linear(x, w_full.to(dtype=x.dtype))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_sketch, latent, freq, phase, sketch = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        lat = latent.view(latent.shape[0], latent.shape[1] // ctx.m, ctx.m).float()
        ang = torch.einsum("obm,mk->obk", lat, freq.float()) + phase.float().view(1, 1, -1)
        w_full = strict_sign(torch.sin(ang)).reshape(latent.shape[0], -1)
        grad_input = grad_flat.matmul(w_full.to(dtype=grad_flat.dtype)).view(ctx.xshape)
        grad_sketch = torch.einsum("no,nir->oir", grad_flat.float(), x_sketch.float())
        grad_weight = torch.einsum("oir,kr->oik", grad_sketch, sketch.float()) / float(sketch.shape[-1])
        grad_latent = torch.einsum("obk,mk->obm", grad_weight * torch.cos(ang), freq.float()).reshape_as(latent)
        return grad_input, grad_latent, None, None, None, None, None


def binary_linear_apply(x: Tensor, module: nn.Module, already_binarized: bool = False, residual_input: Tensor | None = None) -> Tensor:
    if module.impl == "sine":
        x_in = x if already_binarized else binarize(x)
        out = SineLinearFunction.apply(x_in, module.latent, module.freq, module.phase, module.sketch, module.m, module.k)
    else:
        x_in = x if already_binarized else binarize(x)
        out = F.linear(x_in, binary_linear_weight(module.weight, module.threshold).to(dtype=x.dtype))
    out = out * binary_shift_scale(module.shift_param)
    return maybe_add_size_adapter(module, x if residual_input is None else residual_input, out)


def binary_linear_group(x: Tensor, *modules: nn.Module) -> list[Tensor]:
    x_bin = binarize(x)
    return [binary_linear_apply(x_bin, module, already_binarized=True, residual_input=x) for module in modules]


class BinaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, init_std: float = 0.02):
        super().__init__()
        self.impl = os.environ.get("LINEAR_IMPL", "standard")
        gate_rank = int(os.environ.get("SIZE_GATED_RANK", 0))
        self.size_adapter = (
            SizeGatedAdapter(
                in_features,
                out_features,
                rank=gate_rank,
                gate_init=float(os.environ.get("SIZE_GATE_INIT", -6.0)),
                bytes_per_param=float(os.environ.get("SIZE_BYTES_PER_PARAM", 1.0)),
            )
            if gate_rank > 0
            else None
        )
        self.shift_param = nn.Parameter(torch.tensor(-4.5))
        if self.impl == "sine":
            self.k = int(os.environ.get("SINE_K", 16))
            self.m = int(os.environ.get("SINE_M", 1))
            if in_features % self.k != 0:
                raise ValueError(f"sine linear requires in_features={in_features} divisible by k={self.k}")
            self.latent = nn.Parameter(torch.randn(out_features, (in_features // self.k) * self.m) * (0.5 / math.sqrt(max(1, self.m))))
            freq, phase, sketch = make_sine_bank(self.m, self.k, seed=9999 + in_features + out_features)
            self.register_buffer("freq", freq, persistent=False)
            self.register_buffer("phase", phase, persistent=False)
            self.register_buffer("sketch", sketch, persistent=False)
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * init_std)
            self.threshold = nn.Parameter(torch.zeros(out_features, 1))

    def forward(self, x: Tensor) -> Tensor:
        return binary_linear_apply(x, self)


class BinaryWeightLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, init_std: float = 0.02):
        super().__init__()
        self.impl = os.environ.get("LINEAR_IMPL", "standard")
        gate_rank = int(os.environ.get("SIZE_GATED_RANK", 0))
        self.size_adapter = (
            SizeGatedAdapter(
                in_features,
                out_features,
                rank=gate_rank,
                gate_init=float(os.environ.get("SIZE_GATE_INIT", -6.0)),
                bytes_per_param=float(os.environ.get("SIZE_BYTES_PER_PARAM", 1.0)),
            )
            if gate_rank > 0
            else None
        )
        if self.impl == "sine":
            self.k = int(os.environ.get("SINE_K", 16))
            self.m = int(os.environ.get("SINE_M", 1))
            if in_features % self.k != 0:
                raise ValueError(f"sine linear requires in_features={in_features} divisible by k={self.k}")
            self.latent = nn.Parameter(torch.randn(out_features, (in_features // self.k) * self.m) * (0.5 / math.sqrt(max(1, self.m))))
            freq, phase, sketch = make_sine_bank(self.m, self.k, seed=19999 + in_features + out_features)
            self.register_buffer("freq", freq, persistent=False)
            self.register_buffer("phase", phase, persistent=False)
            self.register_buffer("sketch", sketch, persistent=False)
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * init_std)
            self.threshold = nn.Parameter(torch.zeros(out_features, 1))
        self.shift_param = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: Tensor) -> Tensor:
        if self.impl == "sine":
            out = SineLinearFunction.apply(x, self.latent, self.freq, self.phase, self.sketch, self.m, self.k)
        else:
            w_bin = binary_linear_weight(self.weight, self.threshold).to(dtype=x.dtype)
            out = F.linear(x, w_bin)
        out = out * binary_shift_scale(self.shift_param)
        return maybe_add_size_adapter(self, x, out)


class BinaryEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, width_mult: int = 1, extra_layers: int = 1):
        super().__init__()
        self.impl = os.environ.get("EMBED_IMPL", "dense")
        self.hidden_size = hidden_size
        self.inner_size = hidden_size * max(1, int(width_mult))
        self.weight = nn.Parameter(torch.randn(vocab_size, self.inner_size) * 0.02)
        if self.impl == "binary":
            self.threshold = nn.Parameter(torch.zeros(1, self.inner_size))
            self.shift_param = nn.Parameter(torch.tensor(-5.0))
        self.lookup_norm = nn.RMSNorm(self.inner_size)

        if self.inner_size != hidden_size:
            self.proj_norm = nn.RMSNorm(self.inner_size)
            self.proj = BinaryWeightLinear(self.inner_size, hidden_size)
            with torch.no_grad():
                self.proj.shift_param.fill_(-5.0)
        else:
            self.proj_norm = None
            self.proj = None

        self.adapter_norms = nn.ModuleList()
        self.adapter_layers = nn.ModuleList()
        for _ in range(max(0, int(extra_layers))):
            self.adapter_norms.append(nn.RMSNorm(hidden_size))
            layer = BinaryWeightLinear(hidden_size, hidden_size)
            with torch.no_grad():
                layer.shift_param.fill_(-6.0)
            self.adapter_layers.append(layer)

    def binary_lookup_weight(self) -> Tensor:
        if self.impl != "binary":
            return self.weight
        return binarize(self.weight - self.threshold)

    def project(self, hidden_states: Tensor) -> Tensor:
        weight = self.binary_lookup_weight().to(dtype=hidden_states.dtype)
        logits = F.linear(hidden_states, weight)
        return logits if self.impl != "binary" else logits * binary_shift_scale(self.shift_param)

    def forward(self, input_ids: Tensor) -> Tensor:
        weight = self.binary_lookup_weight()
        hidden_states = F.embedding(input_ids, weight)
        if self.impl == "binary":
            hidden_states = hidden_states * binary_shift_scale(self.shift_param)
        hidden_states = self.lookup_norm(hidden_states)
        if self.proj is not None:
            hidden_states = self.proj(self.proj_norm(hidden_states))
        for norm, layer in zip(self.adapter_norms, self.adapter_layers):
            hidden_states = hidden_states + layer(norm(hidden_states))
        return hidden_states


class BinaryLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, use_norm: bool = True):
        super().__init__()
        self.impl = os.environ.get("HEAD_IMPL", "dense")
        self.use_norm = use_norm
        self.norm = nn.RMSNorm(hidden_size) if use_norm else None
        self.proj = CastedLinear(hidden_size, vocab_size, bias=False) if self.impl != "binary" else BinaryWeightLinear(hidden_size, vocab_size)
        if self.impl == "binary":
            with torch.no_grad():
                self.proj.shift_param.fill_(-5.0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return self.proj(hidden_states)
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.norm = nn.RMSNorm(dim)
        self.q_proj = BinaryLinear(dim, dim)
        self.k_proj = BinaryLinear(dim, kv_dim)
        self.v_proj = BinaryLinear(dim, kv_dim)
        self.proj = BinaryWeightLinear(dim, dim)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.mode = os.environ.get("ATTENTION_MODE", "binary_relu2")
        self.margin = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.attn_chunk_size = max(0, int(os.environ.get("ATTN_CHUNK_SIZE", 128)))

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        h = self.norm(x)
        q, k, v = [t.reshape(bsz, seqlen, nh, self.head_dim).transpose(1, 2) for t, nh in zip(binary_linear_group(h, self.q_proj, self.k_proj, self.v_proj), (self.num_heads, self.num_kv_heads, self.num_kv_heads), strict=True)]
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        if self.mode == "flash":
            q = scale_head_axis(apply_rotary_emb(F.rms_norm(q, (q.size(-1),)), cos, sin), self.q_gain)
            k = apply_rotary_emb(F.rms_norm(k, (k.size(-1),)), cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
            y = y + self.margin.to(dtype=y.dtype) * 0.0
        else:
            q = scale_head_axis(apply_rotary_emb(binarize(q), cos, sin), self.q_gain)
            k = apply_rotary_emb(binarize(k), cos, sin)
            v = binarize(v)
            if self.num_kv_heads != self.num_heads:
                repeats = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)
            margin = 2 ** round_ste(self.margin.clamp(-4, 4))
            chunk = self.attn_chunk_size if 0 < self.attn_chunk_size < seqlen else seqlen
            q_pos = torch.arange(seqlen, device=x.device).view(1, 1, seqlen, 1)
            attn_sum = q.new_zeros((bsz, self.num_heads, seqlen, 1))
            y = q.new_zeros((bsz, self.num_heads, seqlen, self.head_dim))
            for start in range(0, seqlen, chunk):
                end = min(start + chunk, seqlen)
                scores = (q @ k[:, :, start:end, :].transpose(-2, -1)) * margin
                scores = scores.masked_fill(torch.arange(start, end, device=x.device).view(1, 1, 1, -1) > q_pos, -65000.0)
                attn_chunk = F.relu(scores + self.head_dim).pow(2)
                attn_sum = attn_sum + attn_chunk.sum(-1, keepdim=True)
                y = y + attn_chunk @ v[:, :, start:end, :]
            y = y / (2 ** round_ste(torch.log2(attn_sum + 1e-6)))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.norm = nn.RMSNorm(dim)
        self.gate = BinaryLinear(dim, hidden)
        self.up = BinaryLinear(dim, hidden)
        self.mid_norm = nn.RMSNorm(hidden)
        self.proj = BinaryWeightLinear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        gate, up = binary_linear_group(h, self.gate, self.up)
        h = gate * up
        return self.proj(self.mid_norm(h))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.mlp_norm = nn.RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_skip = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        x = scale_last_dim(x, self.resid_scale) + scale_last_dim(x0, self.resid_skip)
        attn_out = self.attn(self.attn_norm(x))
        x = x + scale_last_dim(attn_out, self.attn_scale)
        x = x + scale_last_dim(self.mlp(self.mlp_norm(x)), self.mlp_scale)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        binary_embed_width_mult: int,
        binary_embed_layers: int,
        binary_lm_head_norm: bool,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.checkpoint_blocks = bool(int(os.environ.get("CHECKPOINT_BLOCKS", "0")))
        self.tok_emb = BinaryEmbedding(
            vocab_size,
            model_dim,
            width_mult=binary_embed_width_mult,
            extra_layers=binary_embed_layers,
        )
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(model_dim)
        self.lm_head = None if tie_embeddings else BinaryLMHead(model_dim, vocab_size, use_norm=binary_lm_head_norm)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x0 = x
        skips: list[Tensor] = []
        run_block = (lambda block, a, b: checkpoint(lambda aa, bb: block(aa, bb), a, b, use_reentrant=False)) if self.checkpoint_blocks and self.training else (lambda block, a, b: block(a, b))

        for i in range(self.num_encoder_layers):
            x = run_block(self.blocks[i], x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + scale_last_dim(skips.pop(), self.skip_weights[i])
            x = run_block(self.blocks[self.num_encoder_layers + i], x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = self.tok_emb.project(x)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    compile_tweaks = configure_inductor_for_binary_compile(args)
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps = args.grad_accum_steps
    if grad_accum_steps <= 0:
        raise ValueError(f"GRAD_ACCUM_STEPS must be positive, got {grad_accum_steps}")
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        binary_embed_width_mult=args.binary_embed_width_mult,
        binary_embed_layers=args.binary_embed_layers,
        binary_lm_head_norm=args.binary_lm_head_norm,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compile_status = "disabled"
    if args.use_compile:
        try:
            compiled_model = torch.compile(base_model, dynamic=False, fullgraph=args.compile_fullgraph)
            compile_status = f"enabled fullgraph={args.compile_fullgraph}"
        except Exception as exc:
            compiled_model = base_model
            compile_status = f"fallback ({type(exc).__name__}: {exc})"
    else:
        compiled_model = base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    named_params = list(base_model.named_parameters())
    token_weight_name = "tok_emb.weight"
    head_matrix_names = {
        name for name, p in named_params
        if name.startswith("lm_head") and p.ndim == 2
    }
    binary_thresholds = {
        f"{name.removesuffix('.threshold')}.weight": p
        for name, p in named_params
        if name.endswith(".threshold")
    }
    bop_weight_names = set(binary_thresholds) if args.binary_optimizer == "bop" else set()
    excluded_matrix_names = {token_weight_name, *head_matrix_names}
    matrix_params = [
        p
        for name, p in named_params
        if (
            name not in excluded_matrix_names
            and name not in bop_weight_names
            and p.ndim == 2
            and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        )
    ]
    scalar_params = [
        p
        for name, p in named_params
        if (
            name not in excluded_matrix_names
            and (p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
        )
    ]
    token_params = [p for name, p in named_params if name == token_weight_name and name not in bop_weight_names]
    head_params = [p for name, p in named_params if name in head_matrix_names and name not in bop_weight_names]
    bop_params = [p for name, p in named_params if name in bop_weight_names]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_bop = None
    optimizer_tok = None
    if token_params:
        optimizer_tok = torch.optim.Adam(
            [{"params": token_params, "lr": token_lr, "base_lr": token_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
    if bop_params:
        optimizer_bop = BOP(
            bop_params,
            threshold_map={id(p): binary_thresholds[name] for name, p in named_params if name in bop_weight_names},
            adaptivity=args.bop_adaptivity,
            threshold=args.bop_threshold,
        )
        for group in optimizer_bop.param_groups:
            group["base_lr"] = 1.0
    optimizer_muon = None
    if matrix_params:
        optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [opt for opt in (optimizer_tok, optimizer_bop, optimizer_muon, optimizer_scalar) if opt is not None]
    if head_params:
        optimizer_head = torch.optim.Adam(
            [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1 if optimizer_tok is not None else 0, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    if compile_tweaks:
        compile_status = f"{compile_status} tweaks={','.join(compile_tweaks)}"
    log0(f"compile:{compile_status}")
    log0(f"attention_mode:{args.attention_mode}_gqa")
    log0(f"linear_impl:{args.linear_impl} sine_k:{args.sine_k} sine_m:{args.sine_m} binary_optimizer:{args.binary_optimizer}")
    log0(f"embed_impl:{args.embed_impl} head_impl:{args.head_impl}")
    log0(
        f"size_gated_rank:{args.size_gated_rank} size_loss_weight:{args.size_loss_weight} "
        f"size_gate_init:{args.size_gate_init} size_export_threshold:{args.size_export_threshold}"
    )
    log0(f"attn_chunk_size:{args.attn_chunk_size} checkpoint_blocks:{args.checkpoint_blocks}")
    log0(f"attention_heads:num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr if token_params else 0.0} "
        f"head_lr:{args.head_lr if head_params else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(
        f"log_cadence:train_every:{args.train_log_every} "
        f"val_every:{args.val_loss_every} val_progress_every:{args.val_log_every}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def apply_size_loss(loss: Tensor) -> tuple[Tensor, Tensor]:
        if args.size_gated_rank <= 0 or args.size_loss_weight <= 0:
            return loss, torch.zeros((), device=loss.device)
        size_mb = module_expected_size_bytes(base_model) / (1024.0 * 1024.0)
        size_loss = size_mb * args.size_loss_weight
        return loss + size_loss.to(dtype=loss.dtype), size_loss

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                warmup_loss, _ = apply_size_loss(warmup_loss)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (
            args.val_loss_every > 0
            and step % args.val_loss_every == 0
            and (step > 0 or args.validate_at_step_zero)
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            log0(f"validating:step:{step} val_tokens:{val_tokens.numel() - 1} val_batch_size:{args.val_batch_size}")
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log0,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        train_size_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            loss, size_loss = apply_size_loss(loss)
            train_size_loss += size_loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        train_size_loss /= grad_accum_steps

        if optimizer_muon is not None:
            frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
            muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
            for group in optimizer_muon.param_groups:
                group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            msg = (
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
            if args.size_gated_rank > 0:
                size_mb = module_expected_size_bytes(base_model).detach().item() / (1024.0 * 1024.0)
                msg = f"{msg} size_loss:{train_size_loss.item():.4f} size_mb:{size_mb:.4f}"
            log0(msg)

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    export_state = export_state_dict_with_size_gates(base_model, args.size_export_threshold)
    if master_process:
        torch.save(export_state, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(export_state)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    load_state_dict_with_size_gates(base_model, dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log0,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
