# GEPA with a proposer that never generates

GEPA evolves prompts by handing failure traces to a language model and asking it
to write better instructions. Jev cannot write anything — System One models
return typed decisions, not prose — so the usual recipe needs a second,
generative model bolted onto the loop.

This replaces the mutation operator instead, and keeps everything else from the
authors' `gepa` package (Pareto front, minibatch acceptance, candidate pool) via
`custom_candidate_proposer`. **No generative model appears anywhere in the run.**

## Result

Seeded from the naive first-draft criteria, optimising the nine action
descriptions, 2823 Jev calls, **$0.10**, 32 mutations proposed.

| prompt | val acc | test acc | mean margin (test) |
| --- | --- | --- | --- |
| naive seed | 58.7% | 59.6% | +0.239 |
| **GEPA-evolved** | **84.8%** | **72.3%** | **+0.399** |
| hand-tuned (human) | 89.1% | 83.0% | +0.546 |

It closes **86% of the gap to hand-tuning on the selection set, but only 54% on
held-out test** — and it does not beat the human. The val→test drop tells the
story: +0.9 for the seed, +6.1 hand-tuned, **+12.5 evolved**. Pareto selection
runs against the validation set, so val is really training signal, and the
search overfits it harder than a human writing general rules does.

## What it discovered

The naive seed describes each option mechanically ("Move the gripper 3cm in
+x"). These clauses were added by the search:

```
descend:    Never choose this when the fingers are closed.
            Never choose this when the apple is forward of the gripper.
            Never choose this when the gripper is not yet horizontally over the apple.
move_right: Never choose this when the apple is to the left of the gripper.
move_back:  Never choose this when the gripper is at the right height to grip.
```

That last `descend` clause is, almost word for word, the precondition a human
wrote by hand — **rediscovered from failure statistics alone**. The `move_right`
clause attacks the original left/right coin flip, and note it is *asymmetric*:
the search fixed the pair by constraining one side only and left `move_left`
untouched, which a human would be unlikely to do.

The error profile confirms real learning rather than noise. The seed's dominant
failure was descending when it should have been moving — 14 of its 19 test
errors. After evolution, `descend` errors disappear from the top three entirely;
what remains is the opposite mistake, over-moving (`move_forward` 5,
`move_left` 4). It fixed the disease and slightly overshot the cure.

It also produced junk:

```
open_gripper: When the apple is behind the gripper, choose done instead of this.
              When the apple is forward of the gripper, choose done instead of this.
```

Nonsense — redirecting to `done` has nothing to do with where the apple is.
`open_gripper` has only ~10 instances in the set, and with that little evidence
the discriminative search latches onto conditions that merely correlate. This is
the concrete face of the val→test gap.

## How mutation works without generation

The job splits three ways, each part given to whatever can actually do it:

**Evidence (code).** Jev returns the full distribution, so confusion mass over
failures names which two options failed to separate — no model has to infer it
from traces. A set-logic search over 21 conditions read off the state schema
then finds ones that discriminate the pair: a clause survives only if it holds
on every state where the action is right and fails on at least one where it is
not. Top pairs surfaced this way:

```
descend vs open_gripper   move_right vs ascend   descend vs ascend
descend vs move_right     descend vs move_forward
```

**Judgement (Jev).** A `choice` question over the shortlisted repairs picks
which to apply, with the failure digest as state. The mutation decision is
itself a Jev decision.

**Edit (code).** The chosen clause is composed into that component's text.

New wording comes from a grammar — 21 schema-derived conditions × 3 clause
templates (`only` / `never` / `prefer`). Nothing in the grammar encodes which
action is correct; finding the binding is the search problem. The candidate is
the criteria text itself, one component per action, so the artifact stays a
prompt you can read and diff: [`evolved_prompt.txt`](evolved_prompt.txt).

## Scoring

Margin, `p(best acceptable) − p(best unacceptable)`, not 0/1 accuracy. It is
positive exactly when the pick is right, but keeps moving while a candidate is
still wrong, so the search has a gradient to climb where accuracy is a step
function. Accuracy is still reported, since it is what anyone actually cares
about.

The dataset is frozen — states from oracle rollouts, a balanced grid over the
state schema, and hand-written edge cases — each labelled with the *set* of
actions making non-wasteful progress (several can be right at once). Physics
stays out of the search loop.

## Honest reading

The approach works: a large, real gain over the seed for a dime, with no
generative model. But it did not beat hand-tuning, and the reason is structural.
GEPA's power is an LM inventing phrasing nobody enumerated; this searches a
grammar someone authored. Within that grammar it is efficient and its
credit assignment is *better* than a text-only reflector's, because the
distribution states the confusion outright instead of leaving it to be inferred.
Outside the grammar it cannot go, and with thin evidence per component it
overfits.

Sharper levers than more iterations, if this were taken further: more evidence
per component before a clause is allowed to stick, a held-out gate on top of the
Pareto front, and a penalty on clause count to stop components accreting junk.

## Run it

```sh
python3 -m jevbot.evolve.dataset          # rebuild the frozen dataset
python3 -m jevbot.evolve.run --budget 2500
```

`validate.py` will additionally fly the evolved prompt on the robot task, though
that is a separate question from the one measured here.
