# llminspector

Identify **which LLM** an OpenAI-compatible API is actually serving when the
provider name cannot be trusted.

Dishonest resellers swap models. Stealth endpoints hide the lab. This repo
answers that with **independent signals**, not three clones of the same scan.

## Signals (one implementation each)

| Order | Signal | What it proves | What it cannot prove |
| --- | --- | --- | --- |
| 1 | Tokenizer differential (`usage.prompt_tokens` vs a local vocab zoo) | Family / vocabulary | Size or checkpoint inside that vocab |
| 2 | One-token behavioral JSD vs an official-API reference | Checkpoint / serving identity | Family if the reference is wrong |
| 3 | Serving-stack DNA (errors, `finish_reason`, headers, template offset) | Wrapper vs lab | Which weights |
| 4 | Capability / context class | 8B vs 70B when 1–2 collide | Lab identity |

Do not stack extra tokenizer scanners or extra personality classifiers. Same
meter, same variance, no extra vote.

Tokenizer identity is the hard-to-fake layer: it reads the billing meter. A
provider can only hide it by corrupting `usage`, which breaks charging.
Behavioral identity is what catches same-tokenizer swaps (`4o` billed, `4o-mini`
served). Personality / censorship prompts are out of scope; they bend to a
system prompt.

## Status

Pre-alpha. The public repo is initialized; scanners land next.

Intended first implementation:

1. Differential token counting against Hugging Face `tokenizer.json` files
   (including current vocabs: GLM-5.3, Qwen3.8, …).
2. Official-reference behavioral verify for same-family collisions
   (GLM-5.2 vs 5.3, Qwen3.8-27B vs Max).

Prior art we are **not** vendoring wholesale:

- [tokenizer-fingerprint](https://github.com/iSimplifyMe/tokenizer-fingerprint) — tokenizer zoo ranking
- [modelprint](https://github.com/unclecode/modelprint) — plumbing probes
- [llm-fingerprint-detector](https://github.com/ToseaAI/llm-fingerprint-detector) — one-token JSD ([arXiv:2607.10252](https://arxiv.org/abs/2607.10252))

## Middle-context drop

`llminspect middle-drop` repeats one chat-completions body against an
OpenAI-compatible endpoint and checks whether the middle of the prompt
survived. The body is identical on every call. A DROP means `start` and
`end` are PRESENT and `mid` is MISSING. On the known failure the proxy still
reported the same `usage.prompt_tokens` and `usage.input_tokens`, with
`reasoning_tokens` 0, HTTP 200, and `finish_reason` stop.

Pass the endpoint with `--base-url` or `LLMINSPECTOR_BASE_URL`, and the key with
`--api-key` or `LLMINSPECTOR_API_KEY`. There is no default host. The key is never
printed. Request starts are at least 4.2 seconds apart.

    llminspect middle-drop --models glm-5.3 --efforts low --iterations 30

## Pelican

`llminspect pelican` asks for one SVG of a pelican riding a bicycle. Each call
is a single chat completion with no tools. The prompt is fixed. `max_tokens`
is omitted unless you pass a positive `--max-tokens`. Reasoning effort
defaults to `max`.

The runner starts at 4 in-flight calls. A 429 lowers that cap by one, down to
1, waits, and retries the same call. Five calls is the default.

    llminspect pelican --models deepseek-v4.1-flash --iterations 5

That writes `pelican/index.html` with every drawing on one scrollable page and
serves it at `127.0.0.1:8765` until you stop the process. `--no-serve` only
writes the files. The key and the endpoint use the same flags and environment
variables as `middle-drop`.

## Local form

`llminspect ui` opens one page on `127.0.0.1` port `8787`. Enter the API key,
the model, and the base URL, then run pelican, marker, or both.

Pelican uses the fixed one-shot SVG prompt. Each call is one chat completion
with no tools, and `max_tokens` is left unset. Marker is the middle-drop
probe: filler around the start, middle, and end markers. The context size is
that filler length in characters. A DROP means the middle marker was lost
while the start and end were still present.

Each test has its own run count. Both share one concurrency cap, which starts
at 4. A 429 lowers the in-flight cap by one, down to 1, and retries that call.
Pelican runs first, then marker. Reasoning effort defaults to `max`. While a
call is in flight the page shows httpx errors, time to first token, tok/s, and
input, output, reasoning, and cached tokens. tok/s updates while tokens are
still arriving and counts reasoning tokens. Marker says Passed only when every run kept
the middle marker. The form fields and past runs stay in the browser after a
refresh. Stop ends the run, closes the calls still in flight, and leaves the
rows and drawings already on the page. The page stays up until you stop the
process. The key is sent only to the endpoint.

## Install (when packaged)

```bash
pip install -e .
llminspect --help
```

## License

MIT.
