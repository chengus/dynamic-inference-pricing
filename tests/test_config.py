import pytest

from auction_sim.experiments.config import config_summary, validate_config


def test_seed_sets_must_not_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        validate_config({"seeds": {"calibration": [1], "evaluation": [1]}})


def test_static_summary_calls_out_no_simulation():
    summary = config_summary({"name": "tiny", "horizon_s": 60,
                              "seeds": {"calibration": [0], "evaluation": [100]}})
    assert "Static validation only" in summary["note"]
    assert summary["server"]["max_batch_size"] == 32


def test_reserve_candidates_must_meet_declared_marginal_cost():
    with pytest.raises(ValueError, match="marginal cost"):
        validate_config({"economics": {"marginal_cost_per_token": .01},
                         "candidate_grid": {"uniform_auction": {"reserve_per_token": [.005]}}})
