# Jev playground

A dependency-free Python CLI for poking at [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13)
through OpenRouter.

## Jev is not a chat model

Jev is TypeSafe's "System One" decision model. It does not write prose — it
evaluates a piece of **state** against typed **questions** and returns structured
answers with calibrated probabilities. Sending it to `/chat/completions` gets you:

```
typesafe/jev-1.13 is a decisions model and cannot be used with the
chat/completions endpoint. Use the /api/alpha/decisions endpoint instead.
```

So this CLI talks to `POST https://openrouter.ai/api/alpha/decisions`, whose body
is the [TypeSafe System One schema](https://docs.typesafe.ai/api) with an
OpenRouter model slug.

There are three question types (*primitives*):

| Type | Asks | Returns |
| --- | --- | --- |
| `noul` | a yes/no question | probability that the answer is yes |
| `choice` | pick one of N options | the pick, a probability per option, confidence |
| `score` | rate against ordered levels | a weighted score, a probability per level, confidence |

## Setup

```sh
export OPENROUTER=sk-or-...        # OPENROUTER_API_KEY also works
```

Python 3.10+ is the only requirement; the script uses nothing outside the stdlib.

## Use

Confirm the model answers before spending anything (this sends one tiny question,
because decision models are not listed in OpenRouter's `/models` catalogue):

```sh
./jev.py --check
```

Ask it something:

```sh
./jev.py "Help! My payouts have been failing for 3 days." \
    --noul   'urgent: Does this convey urgency?' \
    --choice 'team: Which team should handle this? = billing | technical | sales' \
    --score  'mood: How frustrated is the customer? = Calm | Frustrated | Very angry'
```

```
urgent  (noul)
    ███████████████████████·    96%  yes

team  (choice) -> billing   [confidence 0.90]
    ██████████████████████··    93%  billing
    ██······················     7%  technical
    ························     0%  sales

mood  (score) -> 1.56 / 2  Very angry   [confidence 0.34]
    █████████████···········    56%  2  Very angry
    ███████████·············    44%  1  Frustrated
    ························     0%  0  Calm
```

Note the low confidence on `mood`: the probability mass is split between two
adjacent levels. That second axis is the point of the model — the answer tells
you *what*, the confidence tells you *whether to act*.

### Question syntax

```
--noul   'name: instructions'
--choice 'name: instructions = option | option: rubric | ...'
--score  'name: instructions = lowest level | ... | highest level'
```

All three are repeatable, and answers come back under the names you choose.
Choice options may carry a `: rubric` describing when they apply; score levels
must be listed low to high. Since `:` and `=` and `|` are separators, anything
fiddlier is better written as JSON and passed with `--questions`.

### State

State is a plain string by default, but Jev also accepts JSON objects and
arrays — useful for chat logs and records:

```sh
./jev.py --state-file conversation.json --json-state --questions questions.json
cat ticket.txt | ./jev.py --noul 'refund: Is a refund being requested?'
```

`--questions FILE` takes a raw JSON map of question id to question object, which
gives you everything the API supports (`criteria` on nouls, structured
instructions) without fighting the shorthand.

## Options

| Flag | Meaning |
| --- | --- |
| `--noul`, `--choice`, `--score` | add a question (repeatable) |
| `--questions FILE` | raw JSON questions map (`-` for stdin) |
| `--state-file FILE` | read state from a file (`-` for stdin) |
| `--json-state` | parse the state as JSON and send it structured |
| `--model` | model slug (default `typesafe/jev-1.13`) |
| `--json` | print the raw API response |
| `-v` | print resolved model, token usage and cost to stderr |
| `--check` | probe the model with one cheap question and exit |

`OPENROUTER_BASE_URL` overrides the API root (default `https://openrouter.ai/api`).

## Cost

Jev charges for input only — output is free, since it returns a handful of
typed values rather than generated text. A three-question call over a short
ticket ran 408 input / 69 output tokens for $0.000017.

## Robot arm demo

`jevbot/` has Jev flying a robot arm: every tool call the arm exposes is an
option in one `choice` question, and Jev picks the next action each control
step until the apple is off the table. 6/6 placements, ~$0.001 per pick.
See [`jevbot/README.md`](jevbot/README.md).

![the arm picking up an apple](docs/episode.png)

## Further reading

- [TypeSafe HTTP API reference](https://docs.typesafe.ai/api)
- [Primitives](https://docs.typesafe.ai/primitives) — the three question types in depth
- [Confidence](https://docs.typesafe.ai/confidence) — why it is not the same as probability
- [Known jagged edges in jev-1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
