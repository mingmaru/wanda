"""
Synthetic tests for Phase 0's pure math (no model needed).

Run any of:
    pytest tests/test_phase0_math.py
    python -m pytest tests/test_phase0_math.py -v
    python tests/test_phase0_math.py            # manual runner at the bottom

These exercise compute_mask, per_row_jaccard, and jaccard_stats with
hand-constructed tensors where the answer is known. Cannot catch
model-loading or hook bugs (no GPU here), but catches typos, off-by-one
errors, sign errors, and aggregation mistakes -- exactly the class of
bugs we cannot detect by syntax-checking alone.

predictions.md section 7 ('forbidden moves') says the Phase 0 metric is
locked. These tests are the executable witness that the metric matches
what the docstring claims.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Make phase0 importable when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0 import compute_mask, per_row_jaccard, jaccard_stats


# --------------- compute_mask ---------------

def test_compute_mask_picks_bottom_k_under_uniform_scaler():
    """With s = 1 everywhere, mask picks the bottom-K entries of |W| per row."""
    W_abs = torch.tensor([
        [4.0, 1.0, 3.0, 2.0],   # bottom 2: indices 1 (val 1), 3 (val 2)
        [10.0, 20.0, 5.0, 7.0], # bottom 2: indices 2 (val 5), 3 (val 7)
    ])
    s = torch.ones(4)
    mask = compute_mask(W_abs, s, sparsity=0.5)
    expected = torch.tensor([
        [False, True, False, True],
        [False, False, True, True],
    ])
    assert torch.equal(mask, expected), f"got\n{mask}\nexpected\n{expected}"


def test_compute_mask_scaler_changes_pruned_dim_under_uniform_weights():
    """With |W| = 1, the scaler ordering determines which column is bottom."""
    W_abs = torch.tensor([[1.0, 1.0]])
    # s = [1, 100] -> v = [1, 10] -> bottom is col 0
    m1 = compute_mask(W_abs, torch.tensor([1.0, 100.0]), sparsity=0.5)
    assert torch.equal(m1, torch.tensor([[True, False]]))
    # s = [100, 1] -> v = [10, 1] -> bottom is col 1
    m2 = compute_mask(W_abs, torch.tensor([100.0, 1.0]), sparsity=0.5)
    assert torch.equal(m2, torch.tensor([[False, True]]))


def test_compute_mask_counterexample_from_docstring():
    """The exact counterexample in phase0.py's docstring: |W| = [10, 1],
    s_eng = [1, 4], s_ko = [1, 200]. Spearman(s_eng, s_ko) = 1.0 but
    Wanda's per-row decisions disagree.

    This is the test that 'Spearman of scaler_row alone is the wrong
    metric' -- if it ever fails, the metric drift is back.
    """
    W_abs = torch.tensor([[10.0, 1.0]])
    m_eng = compute_mask(W_abs, torch.tensor([1.0, 4.0]), sparsity=0.5)
    m_ko = compute_mask(W_abs, torch.tensor([1.0, 200.0]), sparsity=0.5)
    # v_eng = [10*1, 1*2]    = [10, 2]      -> bottom-1 = col 1
    # v_ko  = [10*1, 1*14.14] = [10, 14.14] -> bottom-1 = col 0
    assert torch.equal(m_eng, torch.tensor([[False, True]]))
    assert torch.equal(m_ko, torch.tensor([[True, False]]))
    # And the masks should be DIFFERENT, not identical.
    assert not torch.equal(m_eng, m_ko)


def test_compute_mask_sparsity_zero_yields_no_pruning():
    """sparsity=0 means K=0, so mask is all False."""
    W_abs = torch.rand(3, 4)
    s = torch.rand(4)
    mask = compute_mask(W_abs, s, sparsity=0.0)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 0


def test_compute_mask_sparsity_ratio_count():
    """At sparsity = K/d_in, each row has exactly K Trues."""
    d_out, d_in = 5, 10
    W_abs = torch.rand(d_out, d_in)
    s = torch.rand(d_in)
    for sparsity, expected_k in [(0.3, 3), (0.5, 5), (0.7, 7)]:
        mask = compute_mask(W_abs, s, sparsity=sparsity)
        counts = mask.sum(dim=1).tolist()
        assert all(c == expected_k for c in counts), \
            f"sparsity={sparsity}: expected {expected_k} per row, got {counts}"


# --------------- per_row_jaccard ---------------

def test_per_row_jaccard_identical():
    """Identical bool masks -> per-row Jaccard 1.0 everywhere."""
    m = torch.tensor([[True, False, True, False],
                      [False, True, False, True]])
    j = per_row_jaccard(m, m)
    assert torch.allclose(j, torch.ones(2))


def test_per_row_jaccard_disjoint():
    """Equal-size disjoint masks -> Jaccard 0."""
    m_a = torch.tensor([[True, True, False, False]])
    m_b = torch.tensor([[False, False, True, True]])
    j = per_row_jaccard(m_a, m_b)
    assert torch.allclose(j, torch.zeros(1))


def test_per_row_jaccard_half_overlap():
    """Two size-2 sets with 1-element overlap -> Jaccard 1/3."""
    m_a = torch.tensor([[True, True, False, False]])
    m_b = torch.tensor([[True, False, True, False]])
    # inter = {0} = 1; union = {0, 1, 2} = 3; jaccard = 1/3
    j = per_row_jaccard(m_a, m_b)
    assert torch.allclose(j, torch.tensor([1.0 / 3.0]))


def test_per_row_jaccard_empty_mask_returns_one():
    """All-False masks (no pruning) -> Jaccard 1.0 by our convention."""
    m = torch.zeros(2, 4, dtype=torch.bool)
    j = per_row_jaccard(m, m)
    assert torch.allclose(j, torch.ones(2))


def test_per_row_jaccard_mixed_rows():
    """Different rows can have different Jaccards in one call."""
    # Row 0: identical. Row 1: disjoint. Row 2: half-overlap (1/3).
    m_a = torch.tensor([
        [True, False, True, False],
        [True, True, False, False],
        [True, True, False, False],
    ])
    m_b = torch.tensor([
        [True, False, True, False],   # same as m_a row 0
        [False, False, True, True],   # disjoint from m_a row 1
        [True, False, True, False],   # half-overlap with m_a row 2
    ])
    j = per_row_jaccard(m_a, m_b)
    assert torch.allclose(j, torch.tensor([1.0, 0.0, 1.0 / 3.0]))


# --------------- jaccard_stats ---------------

def test_jaccard_stats_known_values():
    """jaccard_stats reports correct mean, percentiles, count."""
    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    stats = jaccard_stats(arr)
    assert math.isclose(stats["mean"], 0.55)
    assert stats["n_jaccards"] == 10
    # numpy uses linear interpolation for percentiles by default.
    # Don't lock to an exact value; just check the right neighborhood.
    assert 0.10 < stats["p10"] < 0.30
    assert 0.45 < stats["p50"] < 0.65
    assert 0.70 < stats["p90"] < 1.0


def test_jaccard_stats_constant_array_has_zero_std():
    """All-equal array -> std 0."""
    arr = np.full(20, 0.42)
    stats = jaccard_stats(arr)
    assert math.isclose(stats["mean"], 0.42)
    assert math.isclose(stats["std"], 0.0, abs_tol=1e-9)
    assert math.isclose(stats["p10"], 0.42)
    assert math.isclose(stats["p90"], 0.42)


# --------------- manual runner ---------------

if __name__ == "__main__":
    import traceback
    tests = sorted(
        (name, fn) for name, fn in globals().items() if name.startswith("test_")
    )
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception:
            print(f"  ERROR {name}:")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
