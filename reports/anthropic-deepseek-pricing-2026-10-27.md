# Anthropic and DeepSeek API pricing notes

**Retrieved:** 2026-10-27 (America/New_York). The official pages below were opened on this date. Where the provider publishes a separate document/update date, it is identified separately; the Anthropic pricing page has no visible document date.

**Scope:** USD per 1 million text tokens for standard direct API model usage. Prices are split into input/cache-read and output token categories; input and output are not a bundled flat per-request rate. This excludes standalone search/fetch and other tools, media-specific charges, subscription/app plans, taxes, and any account-specific credits or negotiated enterprise terms. Prices may change; use live source URLs when refreshing the simulator.

## Anthropic Claude API

The current official pricing table names **Claude Opus 5.5**, **Claude Sonnet 5.5**, and **Claude Haiku 5.5**. These are representative coding/general-purpose and economical options, not the full catalog: the same current table also lists Claude Fable 5.1 at $10 input / $50 output per million tokens. The source table's model IDs are the displayed model names; this note does not infer API version-ID strings.

| Current model | Standard input (cache miss) | Cache hit/read | Output | 5-minute cache write | 1-hour cache write |
|---|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | $4.00 | $0.20 | $20.00 | $5.00 | $8.00 |
| Claude Sonnet 5.5 | $2.00 | $0.10 | $10.00 | $2.50 | $4.00 |
| Claude Haiku 5.5, prompt ≤100k tokens | $0.10 | $0.01 | $0.50 | $0.125 | $0.20 |
| Claude Haiku 5.5, prompt >100k tokens | $0.50 | $0.05 | $2.50 | $0.625 | $1.00 |

All figures are USD / million tokens. Haiku's prompt-length band is per request and counts all input, including cache reads and writes; crossing 100k moves the request's input and output prices to the higher band. The cache read on Opus 5.5 and Sonnet 5.5 is 5% of base input; Haiku is 10% of its band-specific input. Cache writes are 1.25× base input for five minutes and 2× for one hour.

**Published modes and adjustments:** The Batch API is asynchronous and discounts input and output token charges by 50%; Anthropic says batch and prompt-cache discounts can be combined. Fast mode for Opus 5.5 is published at $8 input / $40 output per MTok and cannot be used with Batch. First-party US-only inference for Claude 4.6 and later carries a 1.1× multiplier across token categories; global routing is default. Volume discounts may be available by case-by-case negotiation. No public priority multiplier, public volume threshold/discount schedule, or bidder-price-clearing auction was found on the reviewed pricing docs. The public tariff is usage-metered by model and feature.

**Relevant extra billing:** server-side tools may add charges; for example Anthropic's server web-search tool has a separate per-search charge. Tool schemas and tool-use prompts also consume input tokens. Keep these outside a plain text-token comparison.

## DeepSeek API

The current official pricing table lists API model aliases **`deepseek-v4-pro`** (version DeepSeek-V4-Pro-0813) and **`deepseek-flash`** (version DeepSeek-V4.1-Flash-1017). The latter is the economical option. Legacy `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` names are temporarily routed to V4.1-Flash-1017 at Flash pricing; do not treat them as separate current model tiers. `deepseek-chat` and `deepseek-reasoner` are obsolete for this price comparison.

| Current API model alias | Input cache miss, peak | Input cache hit, peak | Output, peak | Input cache miss, off-peak | Input cache hit, off-peak | Output, off-peak |
|---|---:|---:|---:|---:|---:|---:|
| `deepseek-v4-pro` (V4-Pro-0813) | $1.32 | $0.044 | $3.96 | $0.66 | $0.022 | $1.98 |
| `deepseek-flash` (V4.1-Flash-1017) | $0.30 | $0.006 | $1.20 | $0.15 | $0.003 | $0.60 |

All figures are USD / million tokens. Peak is Monday–Friday 01:00–04:00 UTC and 06:00–10:00 UTC, excluding Chinese public holidays. All other hours, including weekends and Chinese public holidays, are off-peak at half the peak rate. The API pricing page lists cache-hit and cache-miss input rates separately. Disk context caching is enabled by default, has no code/configuration requirement, and is best-effort; cache-hit and cache-miss token counts are reported separately. Reuse requires a matching persisted prefix unit. The docs do not publish a separate cache-write fee or cache-storage fee in the current model pricing table.

DeepSeek says model usage is charged as token count × the applicable rate. No public priority multiplier, public volume-discount schedule, batch discount, or bidder-price-clearing auction was found in the reviewed official pricing docs. Search and Fetch are separate per-successful-request APIs, not text-token prices; the pricing page lists $0.75 per 1,000 successful calls for each. Its note about calls being free in certain integrations is integration-specific and has a stated future change date, so it is excluded from the plain model comparison.

## Auction-simulator comparison boundaries

These public provider prices are the posted per-token charges for model usage, not a bid, reserve, or auction clearing price. The public pricing pages reviewed do not describe customers submitting bids to set a clearing price. Anthropic has a negotiated enterprise/volume pathway but no published general volume curve; DeepSeek publishes time-of-day rates rather than a customer auction. For like-for-like simulator inputs, preserve separate input miss, input cache-read, and output categories and represent DeepSeek peak/off-peak as a time-dependent price schedule. Do not substitute output prices for input prices or assume a single blended rate without a stated workload mix.

## Official sources opened on 2026-10-27

1. Anthropic, [Claude Platform pricing](https://platform.claude.com/docs/en/about-claude/pricing) — live model/cache table, Batch, fast mode, long-context, tools, volume discounts. Retrieved 2026-10-27; no page publication date shown. The page points users to claude.com/pricing for most current pricing.
2. DeepSeek, [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/) — live aliases, current version IDs, per-token cache-miss/cache-hit/output prices, UTC peak schedule. Retrieved 2026-10-27; no page publication date shown.
3. DeepSeek, [Change Log](https://api-docs.deepseek.com/updates/) — document dates: current relevant V4.1-Flash-1017 update dated 2026-10-17; V4.1-Flash release dated 2026-09-10. The Oct 17 entry says API pricing remained unchanged for the update.
4. DeepSeek, [Context Caching](https://api-docs.deepseek.com/guides/kv_cache/) — automatic disk cache, prefix persistence/matching, hit/miss usage fields, best-effort behavior. Retrieved 2026-10-27; no page publication date shown.
