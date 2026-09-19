"""The statistics, checked against hand-computable answers.

This is the part of the repo most able to mislead: an arithmetic slip here does
not crash, it just prints a confident p-value next to a difference that is not
there. So every helper is pinned to a value worked out independently -- Wilson
intervals against published numbers, McNemar against binomial tails small enough
to write down, and the bootstrap against the one case whose answer must be
exactly zero.
"""

from __future__ import annotations

import pytest

from jevopt.compare import (
    Z95,
    boot_means,
    draw_resamples,
    mcnemar_exact,
    percentile_ci,
    wilson,
)

# ------------------------------------------------------------------- Wilson

@pytest.mark.parametrize("successes,n,lo,hi", [
    (50, 100, 0.40383, 0.59617),          # the textbook symmetric case
    (0, 10, 0.0, 0.27753),                # at p = 0 the interval must stay one-sided
    (10, 10, 0.72247, 1.0),
    (1, 89, 0.00199, 0.06093),            # the test-set size these runs actually use
])
def test_wilson_matches_known_intervals(successes, n, lo, hi):
    got_lo, got_hi = wilson(successes, n)
    assert got_lo == pytest.approx(lo, abs=5e-5)
    assert got_hi == pytest.approx(hi, abs=5e-5)


def test_wilson_stays_inside_the_unit_interval_and_brackets_the_estimate():
    """Unlike the normal approximation it cannot run off the end of [0, 1] --
    which is the whole reason it is used at these sample sizes."""
    slack = 1e-9          # the bound at p = 0 or 1 lands on zero only to rounding
    for n in (1, 5, 89, 500):
        for successes in range(n + 1):
            lo, hi = wilson(successes, n)
            assert 0.0 <= lo <= successes / n + slack
            assert successes / n - slack <= hi <= 1.0


def test_wilson_of_nothing_admits_it_knows_nothing():
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_narrows_as_evidence_grows():
    widths = [wilson(n // 2, n)[1] - wilson(n // 2, n)[0] for n in (20, 100, 1000)]
    assert widths == sorted(widths, reverse=True)
    assert Z95 == pytest.approx(1.96, abs=1e-3)


# ------------------------------------------------------------------ McNemar

def _arms(b: int, c: int, both_right: int = 0, both_wrong: int = 0):
    a_correct = [True] * b + [False] * c + [True] * both_right + [False] * both_wrong
    b_correct = [False] * b + [True] * c + [True] * both_right + [False] * both_wrong
    return a_correct, b_correct


@pytest.mark.parametrize("b,c,expected", [
    (10, 0, 0.001953125),        # 2 * 0.5**10
    (6, 0, 0.03125),             # 2 * 0.5**6
    (0, 6, 0.03125),             # two-sided: the direction cannot matter
    (5, 5, 1.0),                 # a perfect tie is no evidence at all
    (1, 1, 1.0),
    (0, 0, 1.0),                 # nothing to test
])
def test_mcnemar_exact_hits_the_binomial_tail(b, c, expected):
    nb, nc, p = mcnemar_exact(*_arms(b, c))
    assert (nb, nc) == (b, c)
    assert p == pytest.approx(expected, abs=1e-12)


def test_concordant_pairs_do_not_change_the_p_value():
    """The power of the paired test comes precisely from throwing these away:
    instances both arms get right say nothing about which arm is better."""
    _nb, _nc, bare = mcnemar_exact(*_arms(6, 0))
    for right, wrong in ((50, 0), (0, 50), (200, 200)):
        nb, nc, padded = mcnemar_exact(*_arms(6, 0, both_right=right, both_wrong=wrong))
        assert (nb, nc) == (6, 0)
        assert padded == bare


def test_mcnemar_never_exceeds_one():
    """min(b, c) tails doubled can overshoot 1 for a near-tie; the report prints
    this number, so it has to remain a probability."""
    for b in range(8):
        for c in range(8):
            _nb, _nc, p = mcnemar_exact(*_arms(b, c))
            assert 0.0 <= p <= 1.0


def test_mcnemar_p_falls_as_the_disagreement_becomes_one_sided():
    ps = [mcnemar_exact(*_arms(b, 0))[2] for b in (4, 6, 10, 14)]
    assert ps == sorted(ps, reverse=True)
    assert ps[0] > 0.05 and ps[-1] < 0.001


# ---------------------------------------------------------------- bootstrap

def test_resamples_are_shared_and_reproducible():
    """Every arm is resampled on the SAME instance indices -- that sharing is
    what makes the bootstrap paired rather than two independent ones."""
    draws = draw_resamples(20, 50, seed=7)
    assert draws == draw_resamples(20, 50, seed=7)
    assert draws != draw_resamples(20, 50, seed=8)
    assert len(draws) == 50
    assert all(len(d) == 20 and all(0 <= i < 20 for i in d) for d in draws)


def test_an_arm_against_itself_has_exactly_zero_difference():
    """The sanity check the whole paired design rests on: identical vectors must
    produce a degenerate interval, not merely a narrow one."""
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    draws = draw_resamples(len(values), 500, seed=0)
    means = boot_means(values, draws)
    diffs = [x - y for x, y in zip(means, means, strict=True)]
    assert percentile_ci(diffs) == (0.0, 0.0)


def test_the_paired_interval_is_reproducible_under_a_seed():
    a = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    b = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

    def interval(seed):
        draws = draw_resamples(len(a), 1000, seed)
        means_a, means_b = boot_means(a, draws), boot_means(b, draws)
        return percentile_ci([x - y for x, y in zip(means_a, means_b, strict=True)])

    assert interval(0) == interval(0)
    lo, hi = interval(0)
    assert lo <= (sum(a) - sum(b)) / len(a) <= hi


def test_boot_means_averages_the_drawn_indices():
    values = [0.0, 10.0, 20.0]
    assert boot_means(values, [[0, 0, 0], [2, 2, 2], [0, 1, 2]]) == [0.0, 20.0, 10.0]


def test_percentile_ci_brackets_the_bulk_of_the_draws():
    xs = [float(i) for i in range(1000)]
    lo, hi = percentile_ci(xs)
    assert lo == 25.0 and hi == 974.0
    assert percentile_ci([3.0]) == (3.0, 3.0)
