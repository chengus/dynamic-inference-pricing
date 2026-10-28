# Auction pricing and provider tariffs — October 27, 2026

## Findings

Luna ran the local synthetic experiment; the parent agent reviewed the code, sources, billing arithmetic, service-quality gates, and exposure denominators. No paid inference APIs were called. The full regression suite passes: **86 tests**.

Results apply to the small configured candidate grids; this is not a comparison against globally optimized posted tariffs.

- Under truthful bidding, first-price hybrid earned **31.0% more than static tiers at low load and 35.8% more at medium load**. Static tiers were the highest-revenue posted-price baseline in these evaluated configurations.
- Uniform auction earned **about half the flat-price baseline revenue** at both loads. It did not outperform posted pricing on revenue here.
- Uniform and first-price hybrid served **100% of offered value** at low and medium load, with identical allocations and completion timelines. The payment difference explains their revenue difference. First-price limit bidders have zero realized surplus when they bid their full value; robustness to shading and budgets remains untested.
- At high load, **none of the tested configurations passed calibration quality constraints**. There is no qualified high-load winner or held-out allocation comparison.

## Controlled experiment

Each load used 600 simulated seconds per seed, calibration seeds 0–2, and five disjoint evaluation seeds 100–104. Server timing/capacity, length distributions, output cap, reservation slack (1.25), patience, and candidate grids were held fixed. Only base arrival rate and its proportional agent MMPP overlay changed. Evaluation cohorts contained 1,829 / 7,378 / 21,895 offered requests at low / medium / high load. High-load evaluation workloads were generated for tariff replay, but no policy qualified for evaluation.

Every selected low/medium candidate passed the unchanged constraints on every evaluation seed: at least 95% fill in each segment, with segment p95 TTFT and end-to-end latency at most twice the pooled calibration-seed low-load reference. These are common thresholds, not exactly matched latency distributions.

### Revenue comparison

The table reports mean common-window revenue relative to flat price = 1.00, and mean fill across five evaluation seeds. Per-seed exposure is the maximum full drain time across evaluated policies, shared by all policies for that seed.

| Policy | Low revenue index | Medium revenue index | Low fill | Medium fill |
|---|---:|---:|---:|---:|
| Uniform auction | 0.503× | 0.502× | 100.00% | 100.00% |
| First-price hybrid | 3.176× | 3.232× | 100.00% | 100.00% |
| Flat posted price | 1.000× | 1.000× | 99.56% | 99.49% |
| Static tiers | 2.424× | 2.379× | 98.42% | 97.82% |
| Time-of-day grid | 1.000× | 1.000× | 99.56% | 99.49% |
| Posted-price controller | 1.297× | 1.556× | 99.45% | 97.64% |

Original policy-specific GPU-hour rates are retained in the raw evidence as diagnostics. They divide by different drain durations: rejected low-bid requests can wait for expiry and lengthen a posted-price run even after useful work finishes. Comparing those rates directly overstates auction uplift. Fixed 600-second exposure is also included as a sensitivity result. No simulation or SLO threshold changed during this analysis correction.

| Comparison | Low uplift | Medium uplift |
|---|---:|---:|
| First-price hybrid vs flat | 217.6% (95% bootstrap interval 205.5%–228.4%) | 223.2% (95% bootstrap interval 217.7%–229.1%) |
| First-price hybrid vs static tiers | 31.0% (95% bootstrap interval 22.8%–38.4%) | 35.8% (95% bootstrap interval 31.6%–40.6%) |

Intervals resample five paired seed outcomes 10,000 times; they describe this small synthetic study. The static-tier comparison is an additional reviewed analysis against the observed strongest posted-price baseline, not a new hyperparameter selection using evaluation seeds.

The ten-minute window lies in the first hourly price slot, so the time-of-day grid is identical to its flat posted quote here; this run does not test a complete diurnal pricing cycle. Large-cohort welfare bounds use total offered value. Serving all offered value attains that bound, but no large-cohort exact scheduling search was performed.

## Current provider pricing

Official direct-API text prices were fetched on October 27, 2026. Units are USD per million tokens. All replay requests fit the short-prompt bands. The models below are representative options; the study does not establish equal model intelligence, output quality, tokenization, or provider latency.

| Provider/model | Mode | Input, uncached | Cached input, context only | Output |
|---|---|---:|---:|---:|
| OpenAI / gpt-6-astra | standard | $10 | $1 | $50 |
| OpenAI / gpt-6.1-sol | standard | $2 | $0.1 | $10 |
| OpenAI / gpt-6.1-terra | standard | $1 | $0.05 | $5 |
| OpenAI / gpt-6-luna | standard | $0.1 | $0.01 | $0.5 |
| Anthropic / Claude Fable 5.1 | standard | $10 | $0.25 | $50 |
| Anthropic / Claude Opus 5.5 | standard | $4 | $0.2 | $20 |
| Anthropic / Claude Sonnet 5.5 | standard | $2 | $0.1 | $10 |
| Anthropic / Claude Haiku 5.5 | standard | $0.1 | $0.01 | $0.5 |
| DeepSeek / deepseek-v4-pro | peak | $1.32 | $0.044 | $3.96 |
| DeepSeek / deepseek-v4-pro | off-peak | $0.66 | $0.022 | $1.98 |
| DeepSeek / deepseek-flash | peak | $0.3 | $0.006 | $1.2 |
| DeepSeek / deepseek-flash | off-peak | $0.15 | $0.003 | $0.6 |

- **OpenAI:** posted input/output and cache categories; Flex/Batch are half standard rates, while the listed Fast tariffs are twice standard. Batch can take up to 24 hours; Flex allows slower service and resource unavailability, so their price discounts are not an equal-interactive-SLO benchmark. [Official OpenAI pricing](https://developers.openai.com/api/docs/pricing), [Batch](https://developers.openai.com/api/docs/guides/batch), [Flex](https://developers.openai.com/api/docs/guides/flex-processing).
- **Anthropic:** separate input/output, cache read/write categories, and a 50% Batch discount. Opus 5.5 Fast is $8 input / $40 output per million; the comparison does not assume that rate applies to Sonnet. Haiku 5.5 rates shown require prompts of at most 100k tokens. [Official Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing).
- **DeepSeek:** separate cache-hit/cache-miss input and output, with off-peak rates half peak. Peak is Monday–Friday UTC 01:00–04:00 and 06:00–10:00, excluding Chinese public holidays; all other times are off-peak. Replay treats peak and off-peak as separate scenarios. [Official DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/).

The reviewed public tariffs are posted prices; they do not describe the simulator’s customer bid-clearing mechanism. Our current model charges one rate for both prompt and output tokens and has no prefix cache. Its $0.001/token synthetic reserve is **$1,000 per million tokens**. The simulated dollar scale has not been calibrated to provider market prices, so simulated revenue/GPU-hour is not a claim of commercial advantage.

### Invoice replay on the same numeric token workload

The medium-load offered cohort averaged 168,574.2 input and 250,041 output tokens per evaluation seed. Applying each tariff to those same quantities gives the following hypothetical bills. These are cache-miss token arithmetic, not measured provider invoices or a same-text/task benchmark.

| Model | Mode | Mean hypothetical bill per cohort |
|---|---|---:|
| gpt-6.1-sol | standard | $2.8376 |
| gpt-6-luna | standard | $0.1419 |
| Claude Sonnet 5.5 | standard | $2.8376 |
| Claude Haiku 5.5 | standard | $0.1419 |
| deepseek-v4-pro | peak | $1.2127 |
| deepseek-v4-pro | off-peak | $0.6063 |
| deepseek-flash | peak | $0.3506 |
| deepseek-flash | off-peak | $0.1753 |

Replay excludes cache hits/writes, tools, taxes, residency uplifts, credits and enterprise discounts. Batch/Flex costs are arithmetic alternatives, not evidence they meet interactive deadlines. Tokenizers differ across providers, so identical numeric counts do not mean identical text. Uniform-completed cohorts are included only where Uniform was actually evaluated; missing high-load results are not zero-dollar invoices.

## Evidence and reproduction

- [Study configuration](../configs/provider-study.yaml)
- [Verified tariff snapshot](provider-pricing-2026-10-27.json)
- [Compact summary](../results/provider-comparison-2026-10-27/summary.json) and [CSV](../results/provider-comparison-2026-10-27/summary.csv)
- [Reviewed paired comparisons](../results/provider-comparison-2026-10-27/reviewed-analysis.json)
- [Tariff replay CSV](../results/provider-comparison-2026-10-27/provider-invoice-replay.csv) and [aggregates](../results/provider-comparison-2026-10-27/provider-invoice-summary.csv)
- [Full compressed seed-level evidence](../results/provider-comparison-2026-10-27/raw-seed-results.json.gz)
- [Revenue chart](../results/provider-comparison-2026-10-27/revenue-comparison.png) and [tariff chart](../results/provider-comparison-2026-10-27/provider-tariff-comparison.png)

```sh
uv run --locked python -m auction_sim.provider_study --config configs/provider-study.yaml --allow-simulation
# Refresh summaries and tariff arithmetic from saved data, without running simulation:
uv run --locked python -m auction_sim.provider_study --reaggregate-existing
```

Before a commercial or strategy-robust conclusion, the model needs separate input/output billing, realistic calibrated values and marginal costs, cache-aware workload traces, full-day price schedules, and experiments with bid shading and budget pacing.
