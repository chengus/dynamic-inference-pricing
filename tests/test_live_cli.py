import pytest

from auction_sim.__main__ import main


def test_serve_requires_explicit_simulation_opt_in():
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--config", "not-read-before-opt-in.yaml"])
    assert exc.value.code == 2


@pytest.mark.parametrize("args", [
    ["--port", "0"], ["--port", "65536"], ["--seed", "-1"],
    ["--speed", "nan"], ["--speed", "inf"], ["--speed", "0"], ["--speed", "0.01"],
])
def test_serve_rejects_invalid_runtime_options_before_starting(args):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--allow-simulation", *args])
    assert exc.value.code == 2
