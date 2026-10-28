"""Keep engine and calibration checks opt-in, including during full pytest runs."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--allow-simulation", action="store_true", default=False,
        help="Explicitly authorize tests that execute the simulator or oracle.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "simulation: executes the engine or calibration")


def pytest_collection_modifyitems(config, items):
    guarded_files = {"test_engine.py", "test_oracle.py", "test_queueing.py"}
    for item in items:
        if Path(str(item.fspath)).name in guarded_files:
            item.add_marker(pytest.mark.simulation)
        if item.get_closest_marker("simulation") and not config.getoption("--allow-simulation"):
            item.add_marker(pytest.mark.skip(reason="Simulation tests require --allow-simulation"))


@pytest.fixture(autouse=True)
def prevent_unguarded_engine_execution(request, monkeypatch):
    """Fail closed if a new, unmarked test tries to execute the actual engine."""
    if not request.config.getoption("--allow-simulation"):
        from auction_sim.engine import Simulator

        def blocked_run(*args, **kwargs):
            pytest.fail("Engine execution requires pytest --allow-simulation")

        monkeypatch.setattr(Simulator, "run", blocked_run)
