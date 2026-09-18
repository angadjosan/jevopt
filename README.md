# Jev playground

A dependency-free Python CLI for poking at [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13)
through OpenRouter.

## Setup

```sh
export OPENROUTER=sk-or-...        # OPENROUTER_API_KEY also works
```

Python 3.10+ is the only requirement; the script uses nothing outside the stdlib.

## Use

```sh
./jev.py "Write a haiku about type inference"
./jev.py --stream --system "Be terse." "Explain Hindley-Milner"
./jev.py -v --temperature 0.2 --max-tokens 256 "..."   # -v prints token usage
cat prompt.txt | ./jev.py                              # prompt from stdin
```

Before spending tokens, confirm the model is live and see what OpenRouter
advertises for it (context length, pricing, supported parameters):

```sh
./jev.py --check
```

If the slug has moved on, `--check` lists the other models under the same
provider. Point `--model` at whichever one you want.

## Options

| Flag | Meaning |
| --- | --- |
| `--model` | model slug (default `typesafe/jev-1.13`) |
| `--system` | system prompt |
| `--stream` | stream the reply token by token |
| `--temperature`, `--max-tokens` | passed through to the API |
| `-v` | print token usage and finish reason to stderr |
| `--check` | look the model up in `/models` and exit |

`OPENROUTER_BASE_URL` overrides the API root, which is how the CLI is tested
against a local mock.

## Note on sandboxed environments

This code has not yet been run against the live API: the environment it was
written in blocks outbound traffic to `openrouter.ai` at the egress proxy
(HTTP 403 on CONNECT). It was instead verified end to end against a local mock
of the OpenRouter endpoints, covering streaming and non-streaming completions,
`--check`, stdin input, and the error paths. Run `./jev.py --check` first from a
network that can reach OpenRouter to confirm the real thing.
