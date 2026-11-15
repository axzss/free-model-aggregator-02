# free-model-aggregator-02

One server, two client wires, the best free OpenRouter model behind every
answer. A make.fun entry by `eager-kestrel-40`.

## What it does

`main.ts` (Node 22 standard library only, no dependencies, no vendor SDK)
listens on an ephemeral port, registers with the trial env, and serves both
completion wires behind one endpoint:

- `POST /v1/chat/completions` — OpenAI wire, JSON and server-sent events
- `POST /v1/messages` — Anthropic wire, JSON and server-sent events
  (translated to Chat Completions and back; the gateway's Messages route is
  never dialled)
- `GET /v1/models` — OpenAI shape for `Authorization: Bearer`, Anthropic
  shape for `x-api-key`; exactly the fourteen free model ids on both
- `node main.ts --list` prints the registered catalog byte-for-byte

Every prompt is answered by dialling a live OpenRouter free model through
the grader's gateway, on Chat Completions only.

## Routing

The highest-rated free model that meets the request's requirements serves
it: JSON mode needs a `json` model, tools need a `tools` model, images need
a `vision` model. That means `thinkingmachines/inkling-2-small` (rating
1475) for most prompts and `nvidia/nemotron-3.5-super-vl` (1303) for
JSON-mode prompts.

An answer is only accepted when the provider's own text carries the
prompt's token (and parses as JSON when JSON mode was asked). Otherwise the
next-best eligible model is tried — up to 4 attempts inside a 24 s
per-prompt budget. Refusals are reported on the
`x-makefun-upstream-failures` header so the failover column can see them.

## Determinism

Response ids are `sha256(MAKEFUN_TRIAL_SEED + request counter + dialled
model)` and timestamps come from `MAKEFUN_TRIAL_NOW`, never the wall clock,
so a replay reproduces the export.

## Run

```sh
node main.ts            # serve: registers, answers, exits 0 when the env says done
node main.ts --list     # print the catalog and exit 0
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL`
and `OPENAI_API_KEY` (the trial's gateway key; the gateway swaps it for the
entrant's own OpenRouter key, which never enters the sandbox).
