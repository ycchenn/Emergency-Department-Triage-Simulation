"""
Unit tests for simulation.py.

Run from the repo root with:
    pytest

or from anywhere with:
    pytest tests/test_simulation.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from simulation import (
    SimConfig,
    Policy,
    severity_weight,
    arrival_rate,
    run_simulation,
    summarize,
    AI_SEV_RISK_WEIGHT,
)

def test_severity_weight_is_highest_for_most_critical():
    weights = [severity_weight(s) for s in range(1, 6)]
    assert weights == sorted(weights, reverse=True)
    assert weights[0] == 5
    assert weights[-1] == 1


def test_risk_score_zero_when_confidence_is_perfect():
    for ai_sev in range(1, 6):
        assert Policy.risk_score(confidence=1.0, ai_sev=ai_sev) == pytest.approx(0.0)


def test_risk_score_matches_weight_table_at_zero_confidence():
    for ai_sev, w in AI_SEV_RISK_WEIGHT.items():
        assert Policy.risk_score(confidence=0.0, ai_sev=ai_sev) == pytest.approx(w)


def test_risk_score_peaks_at_ai_severity_three():
    scores = {s: Policy.risk_score(confidence=0.2, ai_sev=s) for s in range(1, 6)}
    assert max(scores, key=scores.get) == 3

class DummyResource:
    """Minimal stand-in for a simpy.PriorityResource, for threshold tests."""
    def __init__(self, count, capacity):
        self.count = count
        self.capacity = capacity


def test_none_policy_never_triggers_review():
    pol = Policy("none")
    threshold = pol.review_threshold(DummyResource(0, 12))
    assert 1.0 <= threshold


def test_all_policy_always_triggers_review():
    pol = Policy("all")
    threshold = pol.review_threshold(DummyResource(0, 12))
    assert 0.0 > threshold


def test_fixed_policy_returns_constant_tau():
    pol = Policy("fixed", tau_fixed=0.23)
    for count in (0, 6, 12):
        assert pol.review_threshold(DummyResource(count, 12)) == 0.23


def test_dynamic_policy_is_more_lenient_when_idle():
    
    pol = Policy("dynamic", tau_base=0.15, low_util=0.3, high_util=0.85)
    idle_threshold = pol.review_threshold(DummyResource(0, 12))      
    busy_threshold = pol.review_threshold(DummyResource(11, 12))     
    assert idle_threshold < busy_threshold


def test_dynamic_policy_threshold_is_monotonic_in_utilization():
    pol = Policy("dynamic", tau_base=0.15, low_util=0.3, high_util=0.85)
    utils = [0.0, 3, 6, 9, 11]
    thresholds = [pol.review_threshold(DummyResource(u, 12)) for u in utils]
    assert thresholds == sorted(thresholds)

def test_peak_window_has_higher_arrival_rate():
    cfg = SimConfig()
    off_peak_rate = arrival_rate(10, cfg)          
    peak_rate = arrival_rate(300, cfg)             
    assert peak_rate > off_peak_rate
    assert peak_rate == pytest.approx(cfg.base_arrival_rate * cfg.peak_multiplier)

@pytest.fixture
def quick_cfg():
    return SimConfig(sim_minutes=120, n_staff=6)


def test_simulation_processes_every_generated_patient(quick_cfg):
    df = run_simulation(quick_cfg, Policy("fixed", tau_fixed=0.15), seed=42)
    assert len(df) > 0
    assert (df["wait"] >= 0).all()


def test_review_all_never_produces_harmful_misses(quick_cfg):
    df = run_simulation(quick_cfg, Policy("all"), seed=7)
    assert df["harmful_miss"].sum() == 0
    assert (df["reviewed"]).all()


def test_no_review_never_reviews_anyone(quick_cfg):
    df = run_simulation(quick_cfg, Policy("none"), seed=7)
    assert not df["reviewed"].any()


def test_summarize_returns_expected_keys(quick_cfg):
    df = run_simulation(quick_cfg, Policy("fixed", tau_fixed=0.15), seed=1)
    summary = summarize(df, "test policy")
    expected_keys = {
        "policy", "n_patients", "avg_wait", "avg_weighted_wait",
        "avg_wait_critical", "avg_wait_missed_critical",
        "pct_reviewed", "harmful_misses", "total_review_minutes",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["n_patients"] == len(df)


def test_same_seed_is_reproducible(quick_cfg):
    df1 = run_simulation(quick_cfg, Policy("dynamic"), seed=99)
    df2 = run_simulation(quick_cfg, Policy("dynamic"), seed=99)
    assert len(df1) == len(df2)
    assert np.allclose(df1["wait"].values, df2["wait"].values)