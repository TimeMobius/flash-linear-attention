# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Native state transport preserves the single-rank BF16/FP16 recurrence."""

import os
import socket
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.ops.cp import FLACPContext, build_cp_context
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from fla.ops.generalized_delta_rule.dplr.backends import DPLRTileLangBackend
from fla.utils import device


def test_recurrent_cp_context_copy():
    context = FLACPContext(dplr_state_mode='recurrent')
    assert context.copy_for_backward().dplr_state_mode == 'recurrent'
    assert FLACPContext().dplr_state_mode == 'affine'


@pytest.mark.parametrize('mode,layout,group,reason', [
    ('invalid', 'contiguous', None, 'Unknown DPLR CP state mode'),
    ('recurrent', 'zigzag', None, 'contiguous layout and a process group'),
    ('recurrent', 'contiguous', None, 'contiguous layout and a process group'),
])
def test_recurrent_cp_rejects_invalid_context(mode, layout, group, reason):
    x = torch.ones(1, 64, 1, 64, dtype=torch.bfloat16)
    context = FLACPContext(dplr_state_mode=mode, layout=layout, group=group)
    with pytest.raises(ValueError, match=reason):
        chunk_dplr_delta_rule(x, x, x, x, x, -x, cp_context=context)


def test_recurrent_cp_rejects_tilelang_route():
    x = torch.ones(1, 64, 1, 64, dtype=torch.bfloat16)
    context = FLACPContext(dplr_state_mode='recurrent')
    accepted, reason = DPLRTileLangBackend().chunk_dplr_delta_rule_verifier(x, x, x, x, x, -x, cp_context=context)
    assert not accepted and 'recurrent CP state transport' in reason


def _worker(rank, port):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', rank=rank, world_size=4, timeout=timedelta(minutes=3))
    groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
    cases = [
        (2, 'dense', 64, 64, torch.bfloat16, False),
        (4, 'dense', 64, 64, torch.bfloat16, False),
        (4, 'crossing', 64, 64, torch.bfloat16, True),
        (4, 'aligned', 64, 32, torch.float16, False),
        (4, 'dense', 128, 64, torch.bfloat16, True),
        (2, 'crossing', 64, 64, torch.bfloat16, False),
    ]
    try:
        for index, (size, layout, kdim, vdim, dtype, cached) in enumerate(cases):
            group = groups[rank % 2] if size == 2 else dist.group.WORLD
            local_rank = dist.get_rank(group)
            length, heads = size * 256, 2
            torch.manual_seed(20261021 + index)
            shape = (1, length, heads, kdim)
            q = F.normalize(torch.randn(shape, device=device, dtype=dtype), dim=-1)
            k = torch.randn_like(q) * 0.1
            v = torch.randn(1, length, heads, vdim, device=device, dtype=dtype)
            key = F.normalize(torch.randn_like(q), dim=-1)
            a, b = -key, key * torch.randn_like(q).sigmoid()
            gk = -0.003 * torch.rand(shape, device=device, dtype=torch.float32)
            incoming = torch.randn_like(v)
            lengths = [length] if layout == 'dense' else (
                [128, length - 256, 128] if layout == 'crossing' else [length // 2, length // 2]
            )
            cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.long)
            full = [x.detach().requires_grad_() for x in (q, k, v, a, b, gk)]
            options = dict(scale=1., safe_gate=True, chunk_size=64, disable_recompute=cached)
            ref, _ = chunk_dplr_delta_rule(*full, cu_seqlens=None if layout == 'dense' else cu, **options)
            ref_grads = torch.autograd.grad(ref, full, incoming)
            base_context = build_cp_context(cu, group=group, layout='contiguous')
            context = replace(base_context, dplr_state_mode='recurrent')
            assert base_context.dplr_state_mode == 'affine'
            sl = slice(local_rank * 256, (local_rank + 1) * 256)
            local = [x[:, sl].detach().contiguous().requires_grad_() for x in full]
            actual, final_state = chunk_dplr_delta_rule(*local, cp_context=context, **options)
            grads = torch.autograd.grad(actual, local, incoming[:, sl].contiguous())
            assert final_state is None
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, ref[:, sl], rtol=0, atol=0)
            for actual_grad, expected in zip(grads, ref_grads):
                assert torch.isfinite(actual_grad).all()
                torch.testing.assert_close(actual_grad, expected[:, sl], rtol=0, atol=0)
            if rank == 0:
                print(f'recurrent CP case {index}: CP{size}/{layout}/K{kdim}/V{vdim}/{dtype}/cached={cached} exact', flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_recurrent_cp_outputs_and_all_gradients_exact():
    if torch.cuda.device_count() < 4:
        pytest.skip('Four GPUs required')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=4, join=True)
