# jevopt

Prompt optimisation for [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13),
a decision model that returns a typed choice instead of text.

Jev picks one option from a list you define. How well it picks depends heavily
on how each option is described, and writing those descriptions is hand-work.
`jevopt` evolves them for you, from labelled examples — **with no
text-generating model anywhere in the loop.**

```sh
pip install -e .
export OPENROUTER=sk-or-...

jevopt optimize --task jevopt.tasks.triage     # evolve the option descriptions
jevopt compare  --task jevopt.tasks.triage --candidate gepa=runs/triage.results.json
jevopt report   --task jevopt.tasks.triage "run=runs/triage.results.json"
jevopt ask "Payouts have failed for 3 days" --noul 'urgent: Is this urgent?'
```

## The problem this solves

[GEPA](https://github.com/gepa-ai/gepa) evolves prompts by handing failure
traces to a language model and asking it to write something better. Jev cannot
write anything — System One models return typed decisions, not prose — so the
usual recipe needs a second, generative model bolted on.

This replaces only the mutation operator and keeps GEPA's engine (Pareto front,
minibatch acceptance, candidate pool) via its `custom_candidate_proposer` hook.
Mutation splits three ways, each part going to whatever can actually do it:

| step | who | what |
| --- | --- | --- |
| evidence | code | Jev returns a full distribution, so *confusion mass* over failures names which two options failed to separate. A search over conditions read off the state schema finds ones that discriminate them. |
| judgement | **Jev** | a `choice` question picks which shortlisted repair to apply — the mutation decision is itself a Jev decision |
| edit | code | the chosen clause is composed into that option's text |

New wording comes from a grammar — conditions derived from your states, crossed
with `only` / `never` / `prefer` templates. Nothing in it encodes which option is
correct; finding the binding is the search problem. The artifact is readable
prompt text you can diff against the seed.

Scoring uses **margin** (`p(best acceptable) − p(best unacceptable)`) rather than
accuracy: positive exactly when the pick is right, but it keeps moving while a
candidate is still wrong, so the search has something to climb.

## Results

Two tasks, each from a deliberately naive seed. Held-out test accuracy, n=89,
split so no situation appears in more than one split.

| alert triage | naive seed | **evolved** | hand-written | random clauses |
| --- | ---: | ---: | ---: | ---: |
| test accuracy | 68.5% | **87.6%** | 97.8% | 81.1% |

All four columns come from one paired evaluation pass
(`runs/triage.compare.json`), where every arm answered the same instances.
Numbers in the individual run file differ by a point or so: Jev is sampled, so a
separate pass is a separate measurement, and mixing the two sources would
compare arms that never sat the same exam.

On triage it evolved this, from an option description that had said only
"automatically apply the runbook remediation":

> **auto_remediate** — Only choose this when the blast radius is small: tier3, or
> tier2 with a sev3 symptom. Never when `service_tier` is "tier1". Never when
> `known_runbook` is "no".

Those are the real labelling rules, recovered from failure statistics alone.

**Read the caveats.** A search-free control — attach two *random* sound clauses
per option — is the arm to beat, and it is genuinely competitive. On triage the
search wins — better on 8 of 10 seeds, worse on none, sign test p=0.008 — but
its best seed matches the evolved arm outright, and a noise-floor control (the
same prompt entered twice) shows 7 of 15 arms do not separate from it at all. A
careful human still beats the optimiser by 10 points, at p=0.004. See
[`docs/findings.md`](docs/findings.md).

## Defining a task

A task is options, labelled states, and the vocabulary of conditions a clause
may mention. Conditions are derived from the states themselves — no grammar to
hand-write.

```python
from jevopt import Task, derive

instances = [{"state": {"severity": "sev1", "tier": "tier1"},
              "acceptable": ["page_oncall"]}, ...]

def build() -> Task:
    return Task(
        name="triage",
        instructions="Decide what to do with this production alert.",
        options={"page_oncall": "Wake the on-call engineer.", ...},
        instances=instances,
        conditions=derive(instances),
        reference=None,          # optional human-written arm to measure against
    )
```

Point any command at it with `--task your.module`. States must be JSON objects
of low-cardinality values — the grammar enumerates fields, so free text will not
work.

## Commands

| | |
| --- | --- |
| `jevopt optimize` | evolve a task's option descriptions |
| `jevopt baselines` | greedy and random clause controls, no search |
| `jevopt compare` | paired comparison of arms on shared instances — McNemar plus a paired bootstrap |
| `jevopt report` | results JSON to markdown |
| `jevopt ask` | ask Jev noul/choice/score questions straight from the shell |

`compare` exists because unpaired accuracy on this much data has a ±10 point
interval, wider than most differences worth arguing about. Every arm answers the
same instances, so the comparison is paired.

## Layout

```
jevopt/        the tool — no simulator, no domain
  task.py        Task and Condition: options, labelled states, vocabulary
  conditions.py  derive conditions from states
  grammar.py     clause templates and rendering
  evidence.py    gates and discriminative scoring, no model calls
  proposer.py    the mutation operator
  adapter.py     GEPA adapter, margin scoring
  optimize.py baselines.py compare.py report.py ask.py cli.py
  tasks/         triage.py — the worked example
tests/         87 tests, no network
runs/          recorded experiment artifacts
```

## Known characteristics

- **The grammar bounds the search.** It can only say things its templates and
  conditions can express; it will not invent phrasing, which is what a
  generative reflector is for.
- **Thin evidence produces junk.** Options with few positive examples are
  refused outright by the gate rather than given a rule fitted to three cases.
- At equal strength the shortlist tie-breaks alphabetically, which favours
  `Never ...` over the equivalent `Only ...`. Harmless but it makes evolved
  prompts read more negatively than they need to.
- Recorded results in `runs/` were produced by the code as committed.

## Development

```sh
pip install -e ".[dev]"
pytest -q
ruff check .
```
