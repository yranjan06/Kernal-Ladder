# correctness.py — RULE ZERO: verify every kernel against PyTorch before benchmarking.
# Why tolerances, not ==: CPU and GPU sum floats in different orders, so the last
# few fp32 digits differ. That's normal, not a bug. We ask "close enough?", not "equal?".

import torch


def check(name: str, got: torch.Tensor, expected: torch.Tensor,
          atol: float = 1e-3, rtol: float = 1e-3) -> bool:
    if got.shape != expected.shape:                      # shape mismatch is a real bug, fail loud
        print(f"[FAIL] {name}: shape mismatch {got.shape} vs {expected.shape}")
        return False

    got_f, exp_f = got.float(), expected.float()
    ok = torch.allclose(got_f, exp_f, atol=atol, rtol=rtol)
    max_diff = (got_f - exp_f).abs().max().item()        # report the worst error so we can judge severity
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max_abs_diff = {max_diff:.3e}")
    return ok