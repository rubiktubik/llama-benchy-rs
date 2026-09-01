# rust-benchy

`rust-benchy` is a native CLI for benchmarking OpenAI-compatible
`/v1/chat/completions` endpoints. It measures prompt processing, token generation,
time to first response, end-to-end time to first token, and peak generation
throughput across configurable prompt sizes, context depths, and concurrency levels.

It is inspired by the `python-reference/llama-benchy` project in this repository,
but it does **not** load or download a local tokenizer. Prompt sizing is calibrated
against the endpoint's own `usage.prompt_tokens`, and measured prompt throughput
uses the token counts reported by the server.

## Build

```console
cargo build --release
./target/release/rust-benchy --help
```

## Quick start

```console
rust-benchy \
  --base-url http://localhost:8000/v1 \
  --model my-model \
  --pp 512 2048 \
  --tg 32 128 \
  --depth 0 4096 \
  --runs 3 \
  --latency-mode generation
```

`OPENAI_BASE_URL` and `OPENAI_API_KEY` can be used instead of the corresponding
flags. If `--model` is omitted, the CLI selects the model only when `/models`
returns exactly one entry. Use `--served-model-name` when the name sent to the API
differs from the label you want in the report.

Useful options include:

- `--concurrency 1 4 16` for load and aggregate-throughput tests.
- `--enable-prefix-caching` for a separate context-load phase before inference.
- `--exact-tg` to send `min_tokens=<tg>` and `ignore_eos=true` to servers that
  support those extensions.
- `--no-cache` to add a UUID to prompts and send `cache_prompt=false`.
- `--extra-body temperature=0,top_p=1` to add or override JSON request fields.
- `--prompt-file corpus.txt` to avoid downloading the default Project Gutenberg
  corpus.
- `--format md|json|csv` and `--save-result PATH` for machine-readable reports.

Lists accept spaces or commas, so `--pp 512,2048` and `--pp 512 2048` are both
valid.

## Tokenizer-free sizing

Before the suite, rust-benchy sends two short and two long non-streaming probes. It
uses `usage.prompt_tokens` to estimate characters per token and chat-template
overhead separately for user prompts and system context. Corpus slices are then
sized from that calibration. This avoids model-specific client dependencies and
keeps custom OpenAI-compatible model names usable.

The calibration is an estimate because token density varies across text. Metrics
use each response's server-reported `prompt_tokens` when it plausibly describes
the measured phase. In a cached-context inference phase the API usage covers both
the cached context and new prompt, so the calibrated prompt target is used
instead. Target sizes remain the labels used to compare test shapes.
`--no-warmup` disables both calibration and discarded per-shape warmups, using
four characters per token as a fallback.

For completion counts, rust-benchy uses this priority:

1. `choices[0].token_ids` in every stream chunk;
2. final `usage.completion_tokens` from `stream_options.include_usage`;
3. content-bearing SSE chunk count as an explicitly approximate fallback.

This handles multi-token prediction chunks without requiring local tokenization.

## Metrics

- Prompt `t/s`: server-reported input tokens divided by estimated prompt processing
  time (`TTFR - latency baseline`).
- Generation `t/s`: tokens observed after the first token timestamp divided by the
  interval from first to last token. A response delivered in one burst has no
  meaningful decode rate and leaves this field blank.
- `peak t/s`: highest token count observed in a one-second sliding window.
- `ttfr`: time until the first SSE choice event.
- `est_ppt`: TTFR minus the selected latency baseline.
- `e2e_ttft`: time until the first content or reasoning token.

With concurrency greater than one, the report includes both aggregate and
per-request throughput.

## Compatibility notes

The endpoint must support chat completions and streaming SSE. Accurate calibration
requires `usage.prompt_tokens` on non-streaming responses. Accurate completion
counts require either streamed token IDs or streamed final usage. Use
`--skip-coherence` for models that cannot reliably answer the default one-word
sanity check.

Extra fields such as `min_tokens`, `ignore_eos`, `cache_prompt`, and
`return_token_ids` are common extensions, not part of every OpenAI-compatible
implementation. Unknown default fields are normally ignored by compatible servers;
use `--extra-body return_token_ids=false` if a strict server rejects one.
