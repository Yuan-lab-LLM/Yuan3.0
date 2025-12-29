# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused Adam optimizer."""
from __future__ import annotations
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain
from typing import Optional
import warnings
import os

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8Quantizer

import torch
import math
import importlib.util

def detect_cpu_vendor():
    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if line.lower().startswith("vendor_id"):
                return line.split(":")[1].strip()
    return "Unknown"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
if "Intel" in detect_cpu_vendor():
    so_path  = os.path.join(BASE_DIR, "build_adamw",
                                    "adamw_fused_cpu_avx2.cpython-310-x86_64-linux-gnu.so")
    so_path1 = os.path.join(BASE_DIR, "build_adamw",
                                    "cpu_bf16_res_avx2.cpython-310-x86_64-linux-gnu.so")
elif "AMD" in detect_cpu_vendor():
    so_path  = os.path.join(BASE_DIR, "build_adamw",
                                    "adamw_fused_cpu_avx2.cpython-310-x86_64-linux-gnu-amd.so")
    so_path1 = os.path.join(BASE_DIR, "build_adamw",
                                    "cpu_bf16_res_avx2.cpython-310-x86_64-linux-gnu-amd.so")

spec = importlib.util.spec_from_file_location("adamw_fused_cpu_avx2", so_path)
adamw_ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adamw_ext)

spec1 = importlib.util.spec_from_file_location("cpu_bf16_res_avx2", so_path1)
adamw_ext1 = importlib.util.module_from_spec(spec1)
spec.loader.exec_module(adamw_ext1)

def combine_bf16_and_remainder_to_fp32(p_bf16_int16: torch.Tensor,
                                       p_rem_int16: torch.Tensor) -> torch.Tensor:
    # reinterpret raw bits
    hi = p_bf16_int16.to(torch.int32)
    lo = p_rem_int16.to(torch.int32) & 0xFFFF
    i32 = (hi << 16) | lo
    return i32.view(torch.float32)

def bf16_int16_to_bfloat16(x: torch.Tensor) -> torch.bfloat16:
    i32 = (x.to(torch.int32) << 16)
    return i32.view(torch.float32).to(torch.bfloat16)


def bfloat16_to_bf16_int16(x: torch.Tensor) -> torch.int16:
    i32 = x.to(torch.float32).view(torch.int32)
    return (i32 >> 16).to(torch.int16)


def split_fp32_to_bf16_and_remainder(fp32: torch.Tensor):
    i32 = fp32.view(torch.int32)

    # 得到高 16 bit（bf16 位模式）
    p_bf16 = (i32 >> 16).to(torch.int16)

    # 得到低 16 bit（remainder）
    p_rem = (i32 & 0xFFFF).to(torch.int16)

    # CUDA rounding behavior
    # if (rem < 0) bf16++
    neg = (p_rem < 0)
    if neg.any():
        p_bf16 = (p_bf16.to(torch.int32) + neg.to(torch.int32)).to(torch.int16)

    return bf16_int16_to_bfloat16(p_bf16), p_rem


def optimized_step_over_tensor_lists(tensor_lists, adamw_ext, adamw_ext1, beta1, beta2, group_step, eps, lr, weight_decay, store_param_remainders):
    """
    tensor_lists: 原结构，zip(*tensor_lists) yields column_tensor tuples:
      (grad_gpu, param_gpu, exp_avg_cpu, exp_avg_sq_cpu, master_cpu)
    Precondition (from you): column_tensor[0], column_tensor[1], column_tensor[4] are slices from contiguous buffers.
    We will:
      1) build flat GPU view for all grads (zero-copy fast path), else gpu-copy into a big gpu buffer
      2) allocate one pinned host buffer and copy all grads to host once
      3) loop per-param on CPU using host slices for grad + existing exp_avg/exp_avg_sq/master
      4) after CPU updates, copy masters back to GPU in one big H2D op (with dtype cast if needed)
    """
    # build columns
    columns = list(zip(*tensor_lists))  # each element is a tuple (grad_gpu, param_gpu, exp_avg_cpu, exp_avg_sq_cpu, master_cpu)
    if len(columns) == 0:
        return
    # 1) gather grads / params / masters lists (preserve order)
    grads_gpu = [col[0] for col in columns]
    params_gpu = [col[1] for col in columns]
    masters_cpu = [col[4] for col in columns]  # your statement: column[4] is contiguous/slice too


    # helper: try zero-copy flat for a list of contiguous slices
    def try_build_zero_copy_gpu_flat(tensors):
        if len(tensors) == 0:
            return None, False
        dev = tensors[0].device
        dtype = tensors[0].dtype
        # check device/dtype/contiguous
        for t in tensors:
            if t.device != dev or t.dtype != dtype or not t.is_contiguous():
                return None, False
        infos = [(t.data_ptr(), t.numel(), t) for t in tensors]
        elem_size = tensors[0].element_size()

        first = infos[0][2]
        start_ptr = infos[0][0]
        try:
            storage = first.storage()
            base_ptr = storage.data_ptr()
            offset_elems = (start_ptr - base_ptr) // elem_size
            total_elems = sum(n for (_, n, _) in infos)
            flat = torch.empty(0, dtype=dtype, device=dev)
            flat.set_(storage, offset_elems, (total_elems,), (1,))
            return flat, True
        except Exception:
            return None, False
    # try zero-copy for grads and params
    flat_grad_gpu, ok_grads = try_build_zero_copy_gpu_flat(grads_gpu)
    flat_param_gpu, ok_params = try_build_zero_copy_gpu_flat(params_gpu)
    if store_param_remainders:
        flat_master_cpu = None
        ok_master_cpu = False
    else:
        flat_master_cpu, ok_master_cpu = try_build_zero_copy_gpu_flat(masters_cpu)
    # (we won't rely on flat_master_gpu since masters are CPU per your examples)

    # if zero-copy failed for grads -> build a gpu buffer by GPU->GPU copy (fast)
    def build_gpu_flat_by_copy(tensors, dtype=None):
        if len(tensors) == 0:
            return torch.empty(0, device='cuda')
        dev = tensors[0].device
        if dtype is None:
            dtype = tensors[0].dtype
        total = sum(t.numel() for t in tensors)
        flat = torch.empty(total, dtype=dtype, device=dev)
        off = 0
        for t in tensors:
            n = t.numel()
            flat[off:off+n].copy_(t.view(-1))
            off += n
        return flat

    if not ok_grads:
        flat_grad_gpu = build_gpu_flat_by_copy(grads_gpu, dtype=grads_gpu[0].dtype)
        ok_grads = True  # we now have a contiguous GPU buffer

    total_elems = flat_grad_gpu.numel()

    # compute offsets for each column (so we can slice host/gpu big buffers)
    offsets = []
    off = 0
    for g in grads_gpu:
        n = g.numel()
        offsets.append((off, n))
        off += n
    assert off == total_elems

    # 2) allocate one pinned host grad buffer (reuse across steps ideally)
    #    Note: make persistent at object-level in production to avoid re-alloc each step.
    if store_param_remainders:
        param_big_cpu = torch.empty(total_elems, dtype=flat_param_gpu.dtype, device='cpu', pin_memory=True)
        param_big_cpu.copy_(flat_param_gpu, non_blocking=True)
    grad_big_cpu = torch.empty(total_elems, dtype=flat_grad_gpu.dtype, device='cpu', pin_memory=True)
    if flat_grad_gpu.dtype != torch.float32:
        flat_grad_gpu_fp32 = flat_grad_gpu.float()
    else:
        flat_grad_gpu_fp32 = flat_grad_gpu

    grad_big_cpu.copy_(flat_grad_gpu_fp32, non_blocking=True)

    # single H2D copy (device->host) of the whole grad flat buffer
    # non_blocking requires pinned host memory

    torch.cuda.synchronize()  # wait until copy finishes before CPU reads

    # 3) CPU per-parameter processing (your kernel is fast)
    #    For each param use grad slice from grad_big_cpu; exp_avg/exp_avg_sq/master remain as-is (they may be CPU non-contiguous slices)
    for (offset, n), col in zip(offsets, columns):
        grad_slice = grad_big_cpu[offset: offset + n]
        if store_param_remainders:
            param_slice = param_big_cpu[offset: offset + n]
            master_param = adamw_ext1.combine(param_slice, col[4])

        else:
            master_param = col[4]

        exp_avg = col[2]
        exp_avg_sq = col[3]
        # call existing CPU fused kernel (assumed to accept these slices)
        adamw_ext.adamw_fused_cpu_avx2(
            grad=grad_slice,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            master_param=master_param,
            beta1=beta1,
            beta2=beta2,
            step=group_step,
            eps=eps,
            lr=lr,
            weight_decay=weight_decay,
        )
        if store_param_remainders:

            bf16_bits, rem_bits = adamw_ext1.split(master_param)

            param_big_cpu[offset: offset + n].copy_(bf16_bits)
            col[4].copy_(rem_bits)

    # 4) 合并回写：一次性把所有更新后的 master_param (CPU) 转成 target dtype 并拷回 GPU
    #    首先构建 flat_master_cpu by concatenating masters_cpu in same offsets order
    if not ok_master_cpu and not store_param_remainders:
        flat_master_cpu = torch.empty(total_elems, dtype=torch.float32, device='cpu', pin_memory=True)
        off = 0
        for col in columns:
            m = col[4]
            n = m.numel()
            # m may be non-contiguous, but copy into contiguous flat_master_cpu slice
            flat_master_cpu[off:off+n].copy_(m.view(-1))
            off += n
    # if params on GPU are bfloat16 (or other smaller dtype), do a single host->device copy with conversion
    target_device = grads_gpu[0].device
    target_dtype = params_gpu[0].dtype  # e.g. torch.bfloat16
    flat_master_gpu_tmp = torch.empty(total_elems, device=target_device, dtype=target_dtype)
    # host->device (dtype auto-converted by copy_)
    if store_param_remainders:
        flat_master_gpu_tmp.copy_(param_big_cpu, non_blocking=True)
    else:
        flat_master_gpu_tmp.copy_(flat_master_cpu, non_blocking=True)
    torch.cuda.synchronize()

    # now scatter back into param storages (fast GPU->GPU copy or view assignment if params contiguous)
    if ok_params and flat_param_gpu is not None:
        # if params are contiguous and we could create flat_param_gpu, just write in one shot
        # copy into flat_param_gpu (GPU->GPU)
        flat_param_gpu.copy_(flat_master_gpu_tmp)
    else:
        # else do per-param copy (only once per step) from flat_master_gpu_tmp -> each param
        off = 0
        for col in columns:
            p_gpu = col[1]
            n = p_gpu.numel()
            p_gpu.copy_(flat_master_gpu_tmp[off:off+n].view_as(p_gpu), non_blocking=True)
            off += n

    # final sync to ensure/ H2D finished (or schedule to next step)
    torch.cuda.synchronize()
    return


def split_by_numel(all_params, chunk_size):
    """
    all_params: 列表 [(ptr, group_idx, idx, p)]
    根据 p.numel() 按元素数量均匀分成 chunk_size 块（二维列表）
    """

    # 1. 计算总参数量
    total_elems = sum(p.numel() for _, _, _, p in all_params)
    target = total_elems / chunk_size

    chunks = []
    cur_chunk = []
    cur_elems = 0

    for entry in all_params:
        _, _, _, p = entry
        ne = p.numel()

        # 如果放入这个参数会超过目标 chunk 量，并且当前 chunk 不是空的，则开新块
        if cur_elems + ne > target and len(chunks) < chunk_size - 1:
            chunks.append(cur_chunk)
            cur_chunk = []
            cur_elems = 0

        cur_chunk.append(entry)
        cur_elems += ne

    # 最后一块
    if cur_chunk:
        chunks.append(cur_chunk)

    # 保证最终块数 == chunk_size（有可能因为精度问题导致少）
    while len(chunks) < chunk_size:
        chunks.append([])
    
    chunk_elems = [
                    sum(p.numel() for _, _, _, p in chunk)
                            for chunk in chunks
                                ]
    max_chunk_elems = max(chunk_elems) if chunk_elems else 0

    return chunks, max_chunk_elems


def check_param_storage_contiguous(group):
    element_size = None
    infos = []

    # Collect pointer + size information
    # for group in param_groups:
    for p in group["params"]:
        ptr = p.data_ptr()
        size = p.numel()
        esize = p.element_size()

        # All params must have the same dtype to consider global contiguity
        if element_size is None:
            element_size = esize
        elif element_size != esize:
            print("Warning: Found different dtypes; cannot be contiguous globally")

        infos.append((ptr, size, p.shape, esize, p))

    # Sort by pointer (necessary)
    infos.sort(key=lambda x: x[0])

    print("Total tensors:", len(infos))
    print("---- Checking for physical contiguous layout ----")

    contiguous = True
    for i in range(len(infos) - 1):
        ptr, size, shape, esize, p = infos[i]
        next_ptr, next_size, next_shape, next_esize, next_p = infos[i+1]

        expected_next = ptr + size * esize

        if next_ptr != expected_next:
            contiguous = False
            print(f"[NOT CONTIGUOUS] Tensor {i} ({shape}) ends at {expected_next}, "
                  f"but next tensor {i+1} ({next_shape}) starts at {next_ptr}")
        else:
            print(f"[OK] {shape} → {next_shape}")

    if contiguous:
        print("\n>>> All parameters are in one contiguous buffer")
    else:
        print("\n>>> Parameters are NOT stored contiguously (normal in Megatron)")


def check_contiguous_and_linear(tensors):
    """
    检查一列 tensor 是否：
    1. 每个 tensor 自身 contiguous
    2. 对地址排序后，整体是否线性连续排列
    """

    infos = []

    # 先检查自身是否 contiguous，并记录地址和大小
    for t in tensors:
        if not t.is_contiguous():
            print(f"[FAIL] Tensor {t.shape} not contiguous")
            return False

        addr = t.data_ptr()
        size_bytes = t.numel() * t.element_size()

        infos.append((addr, size_bytes, t))

    # ======= 新增：按地址排序 =======
    infos.sort(key=lambda x: x[0])

    # 再检查地址连续性
    for i in range(len(infos) - 1):
        addr, size_bytes, _ = infos[i]
        next_addr, _, _ = infos[i + 1]

        expected_next = addr + size_bytes
        if expected_next != next_addr:
            print(f"[FAIL] Tensor {i} -> {i+1} not continuous in memory")
            print(f"       expected_next={expected_next}, next_addr={next_addr}")
            return False

    print("[OK] All tensors contiguous AND memory is linearly continuous")
    return True


def adamw_single_tensor_cpu_bf16param(
    grad: torch.Tensor,     # GPU FP32
    exp_avg: torch.Tensor,  # CPU FP32
    exp_avg_sq: torch.Tensor,
    master_param: torch.Tensor,
    beta1: float,
    beta2: float,
    step: int,
    eps: float,
    lr: float,
    weight_decay: float,
):

    # ---- 设备 / 类型检查 ----
    assert grad.device.type == "cpu"
    assert grad.dtype == torch.float32
    assert exp_avg.device.type == "cpu"
    assert exp_avg_sq.device.type == "cpu"
    assert master_param.device.type == "cpu"
    assert exp_avg.dtype == exp_avg_sq.dtype == master_param.dtype == torch.float32

    # ---- 1) 从 GPU 拷贝 FP32 版本 ----
    grad_cpu = grad               # FP32

    # master FP32 param 是 AdamW 的主版本
    p_fp32 = master_param

    # ---- 2) 向量化 AdamW 更新 (CPU) ----
    # m = β1*m + (1-β1)*g
    exp_avg.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)

    # v = β2*v + (1-β2)*g*g
    exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)

    # bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    m_hat = exp_avg / bias_correction1
    v_hat = exp_avg_sq / bias_correction2
    denom = torch.sqrt(v_hat) + eps

    update = m_hat / denom

    # weight decay
    if weight_decay != 0:
        update.add_(p_fp32, alpha=weight_decay)

    # 更新 FP32 主参数
    p_fp32.add_(update, alpha=-lr)

    # ---- 3) 写回 GPU，保持 BF16 ----
    # p.copy_(p_fp32.to(torch.bfloat16).cuda())


def get_fp8_meta(fp8_tensor):
    """FP8 metadata getter."""
    assert isinstance(fp8_tensor, Float8Tensor), "Fused optimizer supports only Float8Tensor class"
    if fp8_tensor._quantizer is None:
        raise RuntimeError("FP8 quantizer data is not initialized.")

    quantizer = fp8_tensor._quantizer

    scale = quantizer.scale
    amax = quantizer.amax
    scale_inv = fp8_tensor._scale_inv
    return scale, amax, scale_inv


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Currently GPU-only.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to
        all the model's parameters into one or a few kernel launches.

    :class:`te.optimizers.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = te.optimizers.FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`te.optimizers.FusedAdam` may be used with or without Amp.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        bias_correction (bool, optional): apply correction factor to
            moment estimates. (default: True)
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with CUDA Graphs. (default: False)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16/BF16 mixed precision training.
            (default: False)
        master_weight_dtype (torch.dtype, optional): The dtype of master weights.
            If master_weights is False, this will be ignored. It can be one of
            [torch.float32, torch.float16]. If it's not torch.float32, the optimizer
            will create a FP32 scalar scaling factor to ensure precision.
            (default: torch.float32)
        exp_avg_dtype (torch.dtype, optional): The dtype of exp_avg. It can be
            one of [torch.float32, torch.float16, torch.uint8], where torch.uint8
            represents FP8. If it's not torch.float32, the optimizer will create
            a FP32 scalar scaling factor to ensure precision.
            (default: torch.float32)
        exp_avg_sq_dtype (torch.dtype, optional): The dtype of exp_avg_sq. It
            can be one of [torch.float32, torch.float16, torch.uint8], where
            torch.uint8 represents FP8. If it's not torch.float32, the optimizer
            will create a FP32 scalar scaling factor to ensure precision.
            (default: torch.float32)
        use_decoupled_grad (bool, optional): Whether to use ".decoupled_grad"
            instead of ".grad" for reading gradients. It's useful when the dtypes
            of grad and param are different.
            (default: False)
        store_param_remainders (bool, optional): Whether to store entire FP32 master
            params or just store the trailing 16 remainder bits. Whole FP32 master can be
            reconstructed from BF16 params plus the trailing remainder bits. Works only
            when param type is BF16 and master weight type is FP32, no effect otherwise.
            Useful memory saving optimization.
            (default: False)


    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter | dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        *,
        bias_correction=True,
        adam_w_mode=True,
        capturable=False,
        master_weights=False,
        master_weight_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        use_decoupled_grad=False,
        store_param_remainders=False,
        set_grad_none: Optional[bool] = None,  # deprecated
    ):

        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")

        # Add constraints to dtypes of states.
        if master_weights and master_weight_dtype not in [torch.float32, torch.float16]:
            raise RuntimeError("FusedAdam only supports fp32/fp16 master weights.")
        if exp_avg_dtype not in [torch.float32, torch.float16, torch.bfloat16, torch.uint8]:
            raise RuntimeError("FusedAdam only supports fp32/fp16/bf16/fp8 exp_avg.")
        if exp_avg_sq_dtype not in [torch.float32, torch.float16, torch.bfloat16, torch.uint8]:
            raise RuntimeError("FusedAdam only supports fp32/fp16/bf16/fp8 exp_avg_sq.")

        # Currently, capturable mode only supports fp32 master weights and optimizer states.
        # The reason is, if the master weights or optimizer states are not in fp32 dtype,
        # they will be copied to temporary fp32 buffers first. These fp32 buffers are then
        # used as inputs for the kernel. Consequently, the pointer for earch `.step()` differs,
        # making CUDA Graph inapplicable in this scenario.
        if capturable and master_weights and master_weight_dtype != torch.float32:
            raise RuntimeError("Capturable mode only supports fp32 master weights.")
        if capturable and exp_avg_dtype != torch.float32:
            raise RuntimeError("Capturable mode only supports fp32 exp_avg.")
        if capturable and exp_avg_sq_dtype != torch.float32:
            raise RuntimeError("Capturable mode only supports fp32 exp_avg_sq")
        if capturable and store_param_remainders:
            raise RuntimeError("Capturable mode doesn't support storing param remainders")

        # If the optimizer is capturable then LR should be a tensor (on GPU)
        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0

        self.capturable = capturable
        self.master_weights = master_weights

        if capturable:
            for idx, group in enumerate(self.param_groups):
                if len(group["params"]) == 0:
                    continue
                device = group["params"][0].device
                for item in ["lr"]:
                    self.param_groups[idx][item] = group[item].to(device=device)

            self._step_supports_amp_scaling = True

        # Skip buffer
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self.multi_tensor_adam = tex.multi_tensor_adam
        self.multi_tensor_adam_param_remainder = tex.multi_tensor_adam_param_remainder
        self.multi_tensor_adam_fp8 = tex.multi_tensor_adam_fp8
        self.multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable
        self.multi_tensor_adam_capturable_master = tex.multi_tensor_adam_capturable_master

        self.master_weight_dtype = master_weight_dtype
        self.exp_avg_dtype = exp_avg_dtype
        self.exp_avg_sq_dtype = exp_avg_sq_dtype
        self.name_to_dtype_map = {
            "exp_avg": self.exp_avg_dtype,
            "exp_avg_sq": self.exp_avg_sq_dtype,
            "master_param": self.master_weight_dtype,
        }
        self.dtype_to_range_map = {
            torch.float16: torch.full(
                [1], torch.finfo(torch.float16).max / 2.0, dtype=torch.float32
            ),
            torch.uint8: torch.full([1], 448.0, dtype=torch.float32),
        }
        self._scales = {}
        self.use_decoupled_grad = use_decoupled_grad
        # Works only when master params is in FP32
        self.store_param_remainders = (
            store_param_remainders and master_weights and master_weight_dtype == torch.float32
        )

        # Deprecated options
        self.set_grad_none = set_grad_none
        if self.set_grad_none is not None:
            warnings.warn(
                "set_grad_none kwarg in FusedAdam constructor is deprecated. "
                "Use set_to_none kwarg in zero_grad instead.",
                DeprecationWarning,
            )
        self.unscaled_master_param = torch.zeros(0)
        self.unscaled_exp_avg = torch.zeros(0)
        self.unscaled_exp_avg_sq = torch.zeros(0)

    def zero_grad(self, set_to_none: Optional[bool] = None) -> None:
        """Reset parameter gradients.

        Arguments:
            set_to_none (bool, optional): whether to set grads to `None`
                instead of zeroing out buffers. (default: True)

        """

        # Handle deprecated set_grad_none option
        if self.set_grad_none is not None:
            if set_to_none is not None and set_to_none != self.set_grad_none:
                raise ValueError(
                    f"Called zero_grad with set_to_none={set_to_none}, "
                    f"but FusedAdam was initialized with set_grad_none={self.set_grad_none}"
                )
            set_to_none = self.set_grad_none
        if set_to_none is None:
            set_to_none = True

        if not self.use_decoupled_grad and not set_to_none:
            super().zero_grad(set_to_none=set_to_none)
            return

        for group in self.param_groups:
            for p in group["params"]:
                if self.use_decoupled_grad and set_to_none:
                    p.decoupled_grad = None
                elif self.use_decoupled_grad and not set_to_none:
                    p.decoupled_grad.zero_()
                elif not self.use_decoupled_grad and set_to_none:
                    p.grad = None

    def _apply_scale(self, state_name, unscaled_state, scaled_state, scale):
        """Apply scaling on `unscaled_state`. `scaled_state` and `scale` will be written inplace.

        Arguments:
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            unscaled_state (torch.Tensor): An unscaled high-precision tensor.
            scaled_state (torch.Tensor): An scaled low-precision tensor.
            scale (torch.Tensor): A FP32 tensor representing the scaling factor.
        """
        assert unscaled_state.dtype == torch.float32
        if scaled_state.dtype == torch.bfloat16:
            scaled_state.copy_(unscaled_state.bfloat16())
            return

        dtype = self.name_to_dtype_map[state_name]
        if dtype == torch.uint8:
            assert isinstance(scaled_state, Float8Tensor)
            assert len(scaled_state._quantizer.scale) == 1, (
                "Only scaling with one scaling factor                per tensor is supported by the"
                " FusedAdam."
            )
        else:
            assert scaled_state.dtype == dtype

        max_range = self.dtype_to_range_map[dtype]
        if max_range.device != scaled_state.device:
            max_range = max_range.to(scaled_state.device)
            self.dtype_to_range_map[scaled_state.dtype] = max_range
        if unscaled_state.device != scaled_state.device:
            unscaled_state = unscaled_state.to(scaled_state.device)
        min_val, max_val = torch.aminmax(unscaled_state)
        absmax = torch.maximum(-min_val, max_val)
        absmax = absmax.to(dtype=torch.float32, device=unscaled_state.device)
        torch.div(absmax, max_range, out=scale)
        if isinstance(scaled_state, Float8Tensor):
            scaled_state._quantizer.scale.copy_(1 / scale)
            scaled_state.copy_(unscaled_state)
        else:
            rscale = torch.where(scale > 0, scale.reciprocal(), 0.0)
            unscaled_state.mul_(rscale)
            scaled_state.copy_(unscaled_state)

    def get_unscaled_state(self, param, state_name):
        """Return the unscaled state corresponding to the input `param` and `state_name`.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
        """
        state = self.state[param]
        dtype = self.name_to_dtype_map[state_name]
        if dtype == torch.uint8:
            unscaled = state[state_name].float()
        elif dtype == torch.float16:
            assert state[state_name].dtype == torch.float16
            unscaled = state[state_name].float()
            unscaled.mul_(self._scales[param][state_name])
        elif dtype == torch.float32:
            if (
                self.store_param_remainders
                and state_name == "master_param"
                and param.dtype == torch.bfloat16
            ):
                assert state[state_name].dtype == torch.int16
            else:
                assert state[state_name].dtype == torch.float32
            unscaled = state[state_name]
        elif dtype == torch.bfloat16:
            assert state[state_name].dtype == torch.bfloat16
            unscaled = state[state_name].float()
        else:
            raise RuntimeError(f"Dtype of {state_name} can only be fp8/fp16/bf16/fp32.")
        return unscaled
    
    def get_unscaled_state_master_param(self, param, state_name, unscaled):
        """Return the unscaled state corresponding to the input `param` and `state_name`.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
        """
        state = self.state[param]
        dtype = self.name_to_dtype_map[state_name]
        if dtype == torch.uint8:
            unscaled.copy_(state[state_name].float())
        elif dtype == torch.float16:
            assert state[state_name].dtype == torch.float16

            unscaled.copy_(state[state_name])
            unscaled.mul_(self._scales[param][state_name])
        elif dtype == torch.float32:
            if (
                self.store_param_remainders
                and state_name == "master_param"
                and param.dtype == torch.bfloat16
            ):
                assert state[state_name].dtype == torch.int16
            else:
                assert state[state_name].dtype == torch.float32
            unscaled = state[state_name]
        elif dtype == torch.bfloat16:
            assert state[state_name].dtype == torch.bfloat16
            unscaled = state[state_name].float()
        else:
            raise RuntimeError(f"Dtype of {state_name} can only be fp8/fp16/bf16/fp32.")
        return unscaled

    def set_scaled_state(self, param, state_name, unscaled_state):
        """Set the optimizer state.

        If the dtype of the corresponding optimizer state is not FP32,
        it will do scaling automatically.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            unscaled_state (torch.Tensor): The original high-precision(FP32) state.
        """

        store_param_remainders = (
            self.store_param_remainders
            and state_name == "master_param"
            and param.dtype == torch.bfloat16
        )

        if store_param_remainders:
            assert unscaled_state.dtype == torch.int16
        else:
            assert unscaled_state.dtype == torch.float32
        state = self.state[param]
        if state_name not in state:
            self._initialize_state(param, state_name, False, store_param_remainders)
        dtype = self.name_to_dtype_map[state_name]
        if dtype != torch.float32:
            scale = self._scales[param]
            self._apply_scale(state_name, unscaled_state, state[state_name], scale[state_name])
        else:
            state[state_name].copy_(unscaled_state)

    def _initialize_state(
        self, param, state_name, zero_buffer: bool, store_param_remainders: bool = False
    ):
        """Initialize one of the optimizer states according to `state_name`.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            zero_buffer (bool): Whether to initialize the optimizer state with zeros.
            store_param_remainders (bool): Store only trailing remainder bits.
        """
        dtype = self.name_to_dtype_map[state_name]
        if store_param_remainders:
            data = torch.zeros_like(param, dtype=torch.int16, device="cpu")
        else:
            data = torch.empty_like(param, dtype=dtype, device="cpu")
        if zero_buffer:
            data.zero_()

        if dtype == torch.uint8:
            quantizer = Float8Quantizer(
                scale=torch.ones([1], dtype=torch.float32, device=param.device),
                amax=torch.zeros([1], dtype=torch.float32, device=param.device),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
            self.state[param][state_name] = quantizer.make_empty(param.shape)
            self.state[param][state_name].quantize_(data.float())
        else:

            self.state[param][state_name] = data

        # Create scale if necessary.
        if dtype != torch.float32:
            if param not in self._scales:
                self._scales[param] = {}
            self._scales[param][state_name] = torch.ones(
                [1], dtype=torch.float32, device=data.device
            )

    def initialize_state(self, param, store_param_remainders):
        """Initialize optimizer states.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            store_param_remainders (bool): Store trailing remainder bits.
        """
        self._initialize_state(param, "exp_avg", zero_buffer=True)
        self._initialize_state(param, "exp_avg_sq", zero_buffer=True)
        if self.master_weights:
            self._initialize_state(
                param,
                "master_param",
                zero_buffer=False,
                store_param_remainders=store_param_remainders,
            )
            if not store_param_remainders:
                self.set_scaled_state(param, "master_param", param.clone().detach().float())

    def state_dict(self):
        """Override the state_dict() of pytorch. Before returning the state_dict, cast all
        non-fp32 states to fp32.
        """
        state_dict = super().state_dict()

        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                new_v = {}
                for name in v:
                    new_v[name] = self.get_unscaled_state(param, name)
                state_dict["state"][k] = new_v

        return state_dict

    def load_state_dict(self, state_dict):
        """Override the load_state_dict() of pytorch. Since pytorch's load_state_dict forces the
        state to be the same dtype as param, We need to manully set the state again.
        """
        super().load_state_dict(state_dict)

        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                self.state[param] = {}
                for name in v:
                    if v[name] is None:
                        continue
                    if (
                        self.store_param_remainders
                        and name == "master_param"
                        and param.dtype == torch.bfloat16
                    ):
                        self.set_scaled_state(param, name, v[name])
                        assert v[name].dtype == torch.int16
                    else:
                        self.set_scaled_state(param, name, v[name].float())

    def step(self, closure=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grad_scaler (torch.cuda.amp.GradScaler, optional):
                gradient scaler (default: None)
        """
        loss = None
        if closure is not None:
            loss = closure()
        all_params = []
        device = "cuda"
        bias_correction = 1
        beta1 = 0.90
        beta2 = 0.99
        step = 1

        first_flag = True
        for group_idx, group in enumerate(self.param_groups):
            if len(group["params"]) == 0:
                continue
            if first_flag:
                first_flag = False
                device = group["params"][0].device
                bias_correction = 1 if group["bias_correction"] else 0
                beta1, beta2 = group["betas"]
            else:
                assert device == group["params"][0].device, \
                            f"Device mismatch: device={device}, param_device={group['params'][0].device}"

                assert bias_correction == (1 if group["bias_correction"] else 0), \
                            f"Bias correction mismatch: expected={bias_correction}, group_bias={group['bias_correction']}"

                assert (beta1, beta2) == group["betas"], \
                            f"Betas mismatch: expected={(beta1, beta2)}, group_betas={group['betas']}"


            if "step" in group:
                group["step"] += (
                    1 if not self.capturable else (self._dummy_overflow_buf != 1).to(torch.int)
                )
                step = group["step"]
            else:
                group["step"] = (
                    1 if not self.capturable else torch.tensor([1], dtype=torch.int, device=device)
                )
                step = group["step"]

            for idx, p in enumerate(group["params"]):
                if p is None:
                    continue
                all_params.append((p.data_ptr(), group_idx, idx, p))
        all_params.sort(key=lambda x: x[0])
        chunk_size = 8
        param_chunks, max_chunk_elems = split_by_numel(all_params, chunk_size)
        if self.unscaled_master_param.numel() == 0:
            store_param_remainders = self.store_param_remainders and all_params[0][3].dtype == torch.bfloat16
            if not store_param_remainders:
                self.unscaled_master_param = torch.zeros(max_chunk_elems, dtype=torch.float32, device="cpu")
            self.unscaled_exp_avg = torch.zeros(max_chunk_elems, dtype=torch.float32, device="cpu")
            self.unscaled_exp_avg_sq = torch.zeros(max_chunk_elems, dtype=torch.float32, device="cpu")


        for sub_group in param_chunks:
            # create lists for multi-tensor apply
            p_main_of_fp8_model = []
            p_main_of_f16_model = []
            g_of_fp8_model = []
            g_of_f16_model = []
            g_of_f32_model = []
            m_of_fp8_model = []
            m_of_f16_model = []
            m_of_f32_model = []
            v_of_fp8_model = []
            v_of_f16_model = []
            v_of_f32_model = []
            p_fp8_model = []
            p_f16_model = []
            p_f32_model = []
            # fp8 meta
            scales = []
            amaxes = []
            scale_invs = []

            # Lists for scaling
            unscaled_lists = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}
            scaled_lists = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}
            state_scales = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}

            # Only used when extra params include fp8 tensors. Otherwise, it doesn't matter what the out_dtype is.
            out_dtype = tex.DType.kFloat32

            has_fp16 = False
            has_bf16 = False

            group_offset = 0
            group_total_params = sum(p.numel() for _, _, _, p in sub_group)
            for _, _, _, p in sub_group:
                state = self.state[p]

                store_param_remainders = self.store_param_remainders and p.dtype == torch.bfloat16

                # State initialization
                if len(state) == 0:
                    self.initialize_state(p, store_param_remainders)

                if self.use_decoupled_grad:
                    p_grad = p.decoupled_grad if hasattr(p, "decoupled_grad") else None
                else:
                    p_grad = p.grad

                if p_grad is None:
                    continue
                if p_grad.data.is_sparse:
                    raise RuntimeError("FusedAdam does not support sparse gradients.")

                # Unscaling
                unscaled_state = {}
                for name in ["exp_avg", "exp_avg_sq", "master_param"]:
                    if name in state:
                        if name == "master_param" and store_param_remainders:
                            unscaled_state[name] = self.state[p][name]
                            assert unscaled_state[name].dtype == torch.int16
                        else:
                            if name == "exp_avg":
                                s = slice(group_offset, group_offset + p.numel())
                                unscaled = self.get_unscaled_state_master_param(p, name, self.unscaled_exp_avg[s])
                            elif name == "exp_avg_sq":
                                s = slice(group_offset, group_offset + p.numel())
                                unscaled = self.get_unscaled_state_master_param(p, name, self.unscaled_exp_avg_sq[s])
                            elif name == "master_param":
                                s = slice(group_offset, group_offset + p.numel())
                                unscaled = self.get_unscaled_state_master_param(p, name, self.unscaled_master_param[s])
                            else:
                                unscaled = self.get_unscaled_state(p, name)
                            unscaled_state[name] = unscaled
                        if self.name_to_dtype_map[name] != torch.float32:
                            unscaled_lists[name].append(unscaled)
                            scaled_lists[name].append(state[name])
                            state_scales[name].append(self._scales[p][name])

                if isinstance(p, Float8Tensor):
                    out_dtype = p._fp8_dtype
                    p_fp8_model.append(p._data.data)
                    scale, amax, scale_inv = get_fp8_meta(p)
                    scales.append(scale)
                    amaxes.append(amax)
                    scale_invs.append(scale_inv)
                    if self.master_weights:
                        p_main_of_fp8_model.append(unscaled_state["master_param"].data)
                    g_of_fp8_model.append(p_grad.data)
                    m_of_fp8_model.append(unscaled_state["exp_avg"])
                    v_of_fp8_model.append(unscaled_state["exp_avg_sq"])
                elif p.dtype in [torch.float16, torch.bfloat16]:
                    has_fp16 = has_fp16 or p.dtype == torch.float16
                    has_bf16 = has_bf16 or p.dtype == torch.bfloat16
 
                    p_f16_model.append(p)
                    if self.master_weights:
                        p_main_of_f16_model.append(unscaled_state["master_param"].data)

                    g_of_f16_model.append(p_grad)
                    m_of_f16_model.append(unscaled_state["exp_avg"])
                    v_of_f16_model.append(unscaled_state["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    p_f32_model.append(p.data)
                    g_of_f32_model.append(p_grad.data)
                    m_of_f32_model.append(unscaled_state["exp_avg"])
                    v_of_f32_model.append(unscaled_state["exp_avg_sq"])
                else:
                    raise RuntimeError(
                        "FusedAdam only support model weights in fp32, fp16, bf16 and fp8"
                    )

                if self.capturable and len(p_fp8_model) > 0:
                    raise RuntimeError(
                        "FusedAdam does not support FP8 model weights with capturable=True."
                    )

                if has_fp16 and has_bf16:
                    if self.store_param_remainders:
                        raise RuntimeError(
                            "FusedAdam doesn't support a mix of FP16/BF16 weights + Store param"
                            " remainder."
                        )

                    # simple to add support for this, but not needed for now
                    raise RuntimeError(
                        "FusedAdam does not support a mix of float16 and bfloat16 model weights."
                    )
                group_offset += p.numel()
                def apply_multi_tensor_adam(adam_func, tensor_lists, inv_scale=None, out_dtype=None):
                    # Closures defined in a loop can have unexpected
                    # behavior when called outside the loop. However, this
                    # function is called in the same loop iteration as it
                    # is defined.
                    # pylint: disable=cell-var-from-loop


                    optimized_step_over_tensor_lists(tensor_lists,
                                                     adamw_ext,
                                                     adamw_ext1,
                                                     beta1,
                                                     beta2,
                                                     step,
                                                     group["eps"],
                                                     group["lr"],
                                                     group["weight_decay"],
                                                     self.store_param_remainders
                                                    )



            if self.capturable:
                # If the optimizer is capturable, then if there's a grad scaler it works
                # on the GPU + a different multi_tensor_applier should be called

                # overflow check of gradients
                found_inf = (
                    grad_scaler._check_inf_per_device(self)[device]
                    if grad_scaler is not None
                    else torch.zeros((1,), device=device)
                )
                self._dummy_overflow_buf.copy_(found_inf)

                # get unscale scale factor
                scale, inv_scale = None, None
                if grad_scaler:
                    scale = grad_scaler._get_scale_async()
                    inv_scale = scale.double().reciprocal().float()
                else:
                    scale = torch.ones((1,), device=device)
                    inv_scale = torch.ones((1,), device=device)

                if self.master_weights:
                    if len(p_f16_model) > 0:
                        tensor_lists = [
                            g_of_f16_model,
                            p_f16_model,
                            m_of_f16_model,
                            v_of_f16_model,
                            p_main_of_f16_model,
                        ]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable_master, tensor_lists, inv_scale
                        )
                    if len(p_f32_model) > 0:
                        tensor_lists = [
                            g_of_f32_model,
                            p_f32_model,
                            m_of_f32_model,
                            v_of_f32_model,
                        ]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable, tensor_lists, inv_scale
                        )
                else:
                    if len(p_f16_model) > 0:
                        tensor_lists = [g_of_f16_model, p_f16_model, m_of_f16_model, v_of_f16_model]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable, tensor_lists, inv_scale
                        )
                    if len(p_f32_model) > 0:
                        tensor_lists = [g_of_f32_model, p_f32_model, m_of_f32_model, v_of_f32_model]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable, tensor_lists, inv_scale
                        )

            elif self.master_weights:  # and self.capturable=False
                if len(p_f16_model) > 0:
                    tensor_lists = [
                        g_of_f16_model,
                        p_f16_model,
                        m_of_f16_model,
                        v_of_f16_model,
                        p_main_of_f16_model,
                    ]
                    if self.store_param_remainders and has_bf16 and not has_fp16:
                        # When you have BF16 params and need FP32 master params, you can reconstruct
                        # the FP32 master params with BF16 params + int16 remainders
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_param_remainder, tensor_lists
                        )
                    else:
                        apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
                if len(p_fp8_model) > 0:
                    tensor_lists = [
                        g_of_fp8_model,
                        p_fp8_model,
                        m_of_fp8_model,
                        v_of_fp8_model,
                        p_main_of_fp8_model,
                        scales,
                        amaxes,
                        scale_invs,
                    ]
                    apply_multi_tensor_adam(self.multi_tensor_adam_fp8, tensor_lists, out_dtype)
                if len(p_f32_model) > 0:
                    tensor_lists = [
                        g_of_f32_model,
                        p_f32_model,
                        m_of_f32_model,
                        v_of_f32_model,
                    ]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
            else:  # self.master_weights=False and self.capturable=False
                if len(p_f16_model) > 0:
                    tensor_lists = [g_of_f16_model, p_f16_model, m_of_f16_model, v_of_f16_model]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
                if len(p_f32_model) > 0:
                    tensor_lists = [g_of_f32_model, p_f32_model, m_of_f32_model, v_of_f32_model]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)

            # Scaling
            for name in ["exp_avg", "exp_avg_sq", "master_param"]:
                if len(unscaled_lists[name]) > 0:
                    for unscaled, scaled, scale in zip(
                        unscaled_lists[name], scaled_lists[name], state_scales[name]
                    ):
                        self._apply_scale(name, unscaled, scaled, scale)

            # Try to reclaim the temporary fp32 buffers.
            del unscaled_lists
        # del unscaled_master_param

        return loss
