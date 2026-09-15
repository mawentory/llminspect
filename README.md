# whichllm

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

## Install (when packaged)

```bash
pip install -e .
whichllm --help
```

## License

MIT.
