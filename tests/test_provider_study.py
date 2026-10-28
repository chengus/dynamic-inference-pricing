import json

import pytest

from auction_sim import provider_study
from auction_sim.provider_study import _invoice_replay


def test_study_cli_requires_opt_in_before_running():
    with pytest.raises(SystemExit) as exc:
        provider_study.main([])
    assert exc.value.code == 2


def test_study_cli_accepts_compact_success_report(monkeypatch, capsys):
    monkeypatch.setattr(provider_study, "run_study", lambda *args: {
        "study_design": {"load_labels": ["low", "medium", "high"]}})
    assert provider_study.main(["--allow-simulation"]) == 0
    assert "3 load levels" in capsys.readouterr().out


def test_reaggregation_cli_accepts_compact_report_without_simulation(monkeypatch, capsys):
    monkeypatch.setattr(provider_study, "reaggregate_saved", lambda *args: {
        "policy_summary_rows": [{}] * 18})
    assert provider_study.main(["--reaggregate-existing"]) == 0
    assert "no simulation run" in capsys.readouterr().out


def test_invoice_replay_uses_actual_tokens_and_skips_unrun_served_cohort(tmp_path):
    tariffs = {
        "retrieved_date": "2026-10-27", "currency": "USD", "rate_unit": "per_1m_tokens",
        "replay_assumptions": ["no caching"],
        "rates": [
            {"provider": "OpenAI", "model": "gpt-test", "mode": "standard",
             "input_per_million": 2.0, "output_per_million": 10.0,
             "cached_input_per_million": .1, "source_url": "https://example.com/openai", "notes": "fixture"},
            {"provider": "DeepSeek", "model": "deepseek-test", "mode": "peak",
             "input_per_million": .66, "output_per_million": 1.98,
             "cached_input_per_million": .044, "source_url": "https://example.com/deepseek", "notes": "fixture"},
        ],
    }
    tariffs_path = tmp_path / "tariffs.json"
    tariffs_path.write_text(json.dumps(tariffs))
    summary = {
        "runs": {
            "low": {"policies": {"uniform_auction": {
                "evaluation": [{"seed": 1}],
                "evaluation_request_rows": [[
                    {"status": "completed", "prompt_tokens": 100,
                     "actual_output_tokens": 20, "max_tokens": 999},
                    {"status": "timeout", "prompt_tokens": 500,
                     "actual_output_tokens": 200, "max_tokens": 999},
                ]],
            }}},
            "high": {"policies": {"uniform_auction": {
                "evaluation": [], "evaluation_request_rows": [],
            }}},
        },
        "offered_cohorts_by_load_and_evaluation_seed": {
            "low": {"1": {"all": {"requests": 2, "prompt_tokens": 600,
                                    "actual_output_tokens": 220}}},
            "high": {"2": {"all": {"requests": 1, "prompt_tokens": 300,
                                     "actual_output_tokens": 40}}},
        },
    }
    result = _invoice_replay(summary, tariffs_path, tmp_path)
    rows = result["rows"]
    offered_openai = next(r for r in rows if r["load"] == "low" and r["cohort"] == "offered_all_requests"
                          and r["provider"] == "OpenAI")
    served_openai = next(r for r in rows if r["load"] == "low" and r["cohort"] == "uniform_auction_completed"
                         and r["provider"] == "OpenAI")
    served_deepseek = next(r for r in rows if r["load"] == "low" and r["cohort"] == "uniform_auction_completed"
                           and r["provider"] == "DeepSeek")
    assert offered_openai["request_count"] == 2
    assert offered_openai["billed_usd_uncached"] == (600 * 2 + 220 * 10) / 1_000_000
    assert served_openai["request_count"] == 1
    assert served_openai["billed_usd_uncached"] == (100 * 2 + 20 * 10) / 1_000_000
    assert served_deepseek["billed_usd_uncached"] == (100 * .66 + 20 * 1.98) / 1_000_000
    assert not any(r["load"] == "high" and r["cohort"] == "uniform_auction_completed" for r in rows)
