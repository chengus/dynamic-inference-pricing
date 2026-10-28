# Auction Pricing Simulator

A synthetic discrete-event simulator for dynamic admission pricing of LLM inference requests. It models prefill chunks, batched decode steps, batch and KV reservation limits, queue deadlines, and public order bids. The server timings are explicit synthetic parameters; they are not hardware benchmarks.

## Install and inspect

```sh
uv sync --locked
uv run auction-sim validate --config configs/default.yaml
```

Validation reads the YAML and prints a static summary. It does not create workloads or run the engine. The presets `configs/low.yaml`, `configs/medium.yaml`, and `configs/high.yaml` vary offered synthetic arrival rate, agent burst intensity, and reservation slack while keeping evaluation seeds fixed and separate from calibration seeds. `configs/diurnal.yaml` spans a full 24-hour synthetic price cycle; the one-hour presets only observe the first hourly price slot. `configs/tiny-oracle.yaml` has small cohorts and an enabled, bounded exact oracle for examples.

## Running an experiment

Simulation and calibration are always gated by the explicit `--allow-simulation` flag:

```sh
uv run auction-sim run --config configs/default.yaml --allow-simulation
```

The command constructs paired seeded workloads, measures segment latency references on separate low-load workloads generated from calibration seeds, selects each policy's candidate configuration using calibration seeds subject to every required segment's fill, TTFT, and end-to-end latency SLOs, then evaluates the frozen choice on disjoint seeds. `uv run auction-sim calibrate --config configs/default.yaml --allow-simulation` runs only the calibration phase. If a required segment has no low-load reference completions, the run stops with a clear diagnostic rather than dropping that segment. If no calibration candidate passes all constraints, the report records that policy as infeasible and leaves it unevaluated. Reports preserve seed-level and candidate-level diagnostics in JSON.

For a quick functional check covering all six policies, two calibration seeds, and two evaluation seeds:

```sh
uv run auction-sim run --config configs/smoke.yaml --output results/smoke.json --allow-simulation
```

This preset uses 120 simulated seconds per seed and short capped outputs. The checked run served all 267 evaluation requests, and every policy passed the common service-quality constraints. Busy-time utilization was about 16–19%, so this is a pipeline check rather than a congestion or pricing-effectiveness study. Exact oracle search is skipped above six requests; the report uses the explicitly labeled offered-value welfare bound for these cohorts.

Sweep families can vary arrival load, the reserve shared by all six policies, agent burst shape, or reservation slack. Each creates a deep copy of the input configuration and preserves paired seeds. Burst sweeps adjust the MMPP base rate so the expected agent arrival rate stays fixed as burstiness changes, using the stationary initial-state distribution. Reservation slack changes reservation size while keeping each seed's arrivals, realized outputs, and values fixed.

```sh
uv run auction-sim sweep --config configs/default.yaml --family load --values low,medium,high --allow-simulation
uv run auction-sim sweep --config configs/default.yaml --family reserve --values 0.001,0.004,0.008 --allow-simulation
uv run auction-sim sweep --config configs/default.yaml --family burstiness --values 2,5,8 --allow-simulation
uv run auction-sim sweep --config configs/default.yaml --family reservation_slack --values 1,1.25,1.5 --allow-simulation
```

Revenue per GPU-hour uses the simulator's full elapsed time, including cohort drain after the arrival horizon. Utilization includes busy time, time-weighted batch occupancy, decoding slots, and reserved KV capacity. Latency summaries cover completed requests; fill, timeout, infeasible counts/rates, and admission wait summaries retain failures. TPOT is a diagnostic metric. SLOs use request-level fill and segment p95 TTFT/end-to-end limits derived from an affordable, low-load FIFO reference at twice its pooled calibration-seed latency. Zero completions fail the latency SLO.

Exact hindsight welfare is bounded to small cohorts and is only computed when `oracle.enabled` is true in configuration. The supplied presets enable it: large cohorts immediately use a certified, loose upper bound equal to total offered value. All selected policies on an evaluation seed share the same oracle, searched over engine events plus the union of their admission timestamps. Exactness is scoped to that decision calendar, rather than unrestricted continuous-time scheduling. For a non-exact oracle result, reports distinguish a certified ratio against its supplied upper bound from an exact optimum ratio. No heuristic is labeled an upper bound.

The optional `economics.marginal_cost_per_token` is a synthetic marginal-cost estimate. Candidate reserves must meet that floor. If it is omitted, the report labels cost as unspecified and applies no economic floor. The low-load reference uses a near-zero reserve only to normalize latency; it does not define the economic floor.

## Provider comparison study

The [reviewed October 27, 2026 comparison](reports/provider-comparison-2026-10-27.md)
contains the controlled low/medium/high-load results, official OpenAI/Anthropic/
DeepSeek tariffs, invoice replay, and limitations. The comparison uses common
per-seed observation windows to avoid bias from policy-specific queue drain.
Provider invoices reprice identical numeric token counts; no paid API calls are
made, and provider model quality or latency is not measured.

```sh
uv run --locked python -m auction_sim.provider_study --config configs/provider-study.yaml --allow-simulation
# Recompute summaries from existing saved data, without another simulation:
uv run --locked python -m auction_sim.provider_study --reaggregate-existing
```

## Local live order book

Start the local dashboard against one continuously running, wall-clock-paced
Python simulation:

```sh
uv run auction-sim serve --config configs/live.yaml --port 8765 --seed 42 --allow-simulation
```

Open `http://127.0.0.1:8765`. The light, buildless website polls the local server;
no Node dependencies or hosted service are needed. Pause and resume preserve the
current simulation, and speed controls change its wall-clock pace. Restart
replays the same seed. Changing the payment mechanism starts a fresh run.

The input and output panels show **the same pending requests and bundled bids**.
Input depth uses prompt tokens; output depth uses reserved `max_tokens`, not
future realized output. Price levels aggregate buy-side demand only. The reserve
is the provider's floor, not a seller ask, and the two panels are not independent
markets. Clearing-price history records actual admissions, and admitted work
keeps its original rate. Completed and timed-out requests leave the pending book.

`configs/live.yaml` is a deliberately congested synthetic demonstration so that
standing depth is visible. It runs a single seed for ten simulated minutes plus
drain; it does not run the calibration/evaluation harness. The page identifies
itself as a simulation and reports completion when the cohort has drained.

## Plot saved results

Plotting consumes an existing JSON report and never runs simulations:

```sh
uv run --extra plots auction-sim plot --input results/synthetic-medium-load.json --output results/revenue.png
```

Use `uv sync --locked --extra plots` to install the plotting dependencies. The lockfile records runtime, development, and plotting dependencies; Python 3.12 is selected by `.python-version`.

## Scope and limitations

Demand is synthetic: arrivals use a split diurnal nonhomogeneous Poisson arrivals for `interactive` and `agent` segments, with an additive two-state MMPP burst overlay for `agent` requests. Prompt length, actual output length, and private value use seeded lognormal draws. High-value `interactive` requests are market orders with no queue wait; lower-value `agent` requests are limit orders with a bounded wait. Segment-specific length and value distributions are configurable. The public output cap is applied before reservation slack, so slack changes reserve size without changing arrival times, realized outputs, or values. Requests are bounded and truncated at configured caps. These are workload controls, not claims about a real provider's trace distribution.

Presets include two auction mechanisms and four baselines. Candidate grids are deliberately small and visible in YAML. The simulator does not yet model user strategic bidding, cancellation, token-level billing disputes, multi-GPU placement, network delays, preemption, or adaptive SLO targets. Results should be interpreted as controlled synthetic comparisons.

The complete validation suite passes: **86 tests passed**, including engine timelines, the bounded oracle, a seeded M/M/1 queueing check, public live snapshots, and local HTTP controls. The observer is checked to leave simulated outcomes unchanged. Pure checks cover deterministic demand properties, metric accounting, configuration validation, CLI guards, and harness wiring with fixed result fixtures. The authorized quick experiment completed successfully; larger research experiments have not been run. The dashboard's pause, resume, speed, restart, and mechanism controls were also checked in the browser.

Run pure checks with `uv run pytest`. Engine and oracle tests are skipped by default, and an additional test guard blocks unmarked attempts to execute the engine. Use `uv run pytest --allow-simulation` to include those checks, and `uv run ruff check src tests` for lint checks. uv uses the project's isolated `.venv` rather than globally installed test plugins. Add `--locked` to `uv run` commands to require the saved dependency resolution without updating it.

`clearing_price_series` contains auction clearing prices only. Posted-price and tiered admissions are recorded in `admission_price_series`; a tiered batch's average rate is identified as a mean admission price. The simulator locks a rate, rather than a final invoice: billing uses actual prompt-plus-output tokens after completion.
