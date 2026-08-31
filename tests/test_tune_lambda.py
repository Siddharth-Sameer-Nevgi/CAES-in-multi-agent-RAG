"""Grid resolution and the iteration-spread guard.

The sweep decides the single number the whole contribution hangs on, so the
grid and the degeneracy checks are worth pinning down. See [D-21].
"""
from __future__ import annotations

import pytest

import config
from tune_lambda import (
    COARSE_GRID,
    REFINE_MULTIPLIERS,
    SPREAD_WARN_SHARE,
    f1_is_flat,
    format_distribution,
    format_row,
    iteration_distribution,
    max_bucket_share,
    refine_grid,
    spread_is_degenerate,
)


def Q(*iterations):
    return [{"iterations_used": i} for i in iterations]


# --- grid resolution ------------------------------------------------------

def test_coarse_grid_steps_are_at_most_half_a_decade():
    """A decade-spaced grid steps clean over the gate's sensitive band."""
    for lo, hi in zip(COARSE_GRID, COARSE_GRID[1:]):
        assert hi / lo <= 3.5, f"gap {lo}->{hi} is wider than half a decade"


def test_coarse_grid_is_sorted_and_positive():
    assert COARSE_GRID == sorted(COARSE_GRID)
    assert all(x > 0 for x in COARSE_GRID)


def test_coarse_grid_brackets_the_expected_live_region():
    """Per-iteration cost ~$0.0018 and dQ ~0.03-0.15 puts the knee near 15-80."""
    assert min(COARSE_GRID) < 15 < max(COARSE_GRID)
    assert min(COARSE_GRID) < 80 < max(COARSE_GRID)
    assert any(10 <= x <= 100 for x in COARSE_GRID)


def test_refine_grid_is_log_spaced_about_the_centre():
    centre = 30.0
    grid = refine_grid(centre)
    assert centre in grid
    below = [x for x in grid if x < centre]
    above = [x for x in grid if x > centre]
    assert below and above, "refinement must probe both sides of the knee"
    # Log-spaced means the ratios are symmetric about the centre, not the gaps.
    assert min(above) / centre == pytest.approx(centre / max(below), rel=0.35)


def test_refine_grid_scales_with_the_centre():
    """Multiplicative refinement, so a knee at 3 is probed as finely as one at 300."""
    small, large = refine_grid(3.0), refine_grid(300.0)
    assert [x / 3.0 for x in small] == pytest.approx([x / 300.0 for x in large])
    assert REFINE_MULTIPLIERS[0] < 1.0 < REFINE_MULTIPLIERS[-1]


# --- iteration spread -----------------------------------------------------

def test_distribution_covers_every_bucket_even_when_empty():
    dist = iteration_distribution(Q(2, 2, 3))
    assert set(dist) == set(range(1, config.MAX_ITERATIONS + 1))
    assert dist[1] == 0 and dist[2] == 2 and dist[3] == 1


def test_format_distribution_is_stable_and_csv_safe():
    s = format_distribution(iteration_distribution(Q(2, 2, 3)))
    assert s == "1:0|2:2|3:1|4:0|5:0"
    assert "," not in s and "\n" not in s


def test_max_bucket_share():
    assert max_bucket_share(iteration_distribution(Q(2, 2, 2, 2))) == 1.0
    assert max_bucket_share(iteration_distribution(Q(1, 2, 3, 4))) == 0.25
    assert max_bucket_share({i: 0 for i in range(1, 6)}) == 0.0


def test_all_one_bucket_is_degenerate():
    row = {"max_bucket_share": max_bucket_share(iteration_distribution(Q(*[2] * 40)))}
    assert spread_is_degenerate(row)


def test_genuine_spread_is_not_degenerate():
    dist = iteration_distribution(Q(*([2] * 31 + [3] * 9)))
    assert not spread_is_degenerate({"max_bucket_share": max_bucket_share(dist)})


def test_degeneracy_threshold_boundary():
    assert spread_is_degenerate({"max_bucket_share": SPREAD_WARN_SHARE})
    assert not spread_is_degenerate({"max_bucket_share": SPREAD_WARN_SHARE - 0.01})


# --- the two guards are independent ---------------------------------------

def test_spread_and_f1_guards_are_orthogonal():
    """A lambda can be optimal on F1-per-dollar and useless on spread."""
    rows = [{"f1": 0.40, "mean_usd": 0.005}, {"f1": 0.55, "mean_usd": 0.004}]
    assert not f1_is_flat(rows)                       # quality does respond
    assert spread_is_degenerate({"max_bucket_share": 1.0})   # spread does not


def test_format_row_flags_single_bucket_inline():
    row = {"lambda": 200.0, "mean_iterations": 2.0, "mean_usd": 0.0035,
           "f1": 0.5, "exact_match": 0.4, "iteration_dist": "1:0|2:40|3:0|4:0|5:0",
           "max_bucket_share": 1.0}
    out = format_row(row)
    assert "single bucket" in out
    assert "1:0|2:40" in out

    row["max_bucket_share"] = 0.5
    assert "single bucket" not in format_row(row)


def test_coarse_grid_brackets_the_measured_sensitive_band():
    """The grid must span where the gate's decision actually changes.

    Measured on calibration trajectories (2026-08-31): the iteration histogram
    is identical for every lambda from 1 to 300, degenerate at 1000, and first
    splits at 3000. A grid ceiling of 1000 would step clean over that band and
    report a degenerate lambda as optimal -- the [D-21] failure, one decade up.
    """
    from tune_lambda import COARSE_GRID

    assert max(COARSE_GRID) >= 10000, (
        "grid ceiling is below the measured sensitive band (~1000-10000); "
        "see DECISIONS [D-28]")
    band = [x for x in COARSE_GRID if 1000 <= x <= 10000]
    assert len(band) >= 3, f"only {len(band)} grid points in the sensitive band"
