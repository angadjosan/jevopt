# Findings

Alert triage, optimised from a deliberately naive seed. Held-out test set of 89
instances, split by situation so nothing in test appears in train. Every figure
comes from one paired pass in which all sixteen arms answered the same
instances. The raw artifacts are not kept in the repo; the figures below are
what they recorded.

## Headline

| | naive seed | **evolved** | hand-written | random clauses (10 seeds) |
| --- | ---: | ---: | ---: | ---: |
| test accuracy | 68.5% | **87.6%** | 97.8% | 81.1% (71.9–87.6) |

The evolved prompt beats its naive seed by 19 points (0 vs 17 discordant
instances, p < 0.0001). That is the one result here that is unambiguous.

The per-run file reported the same arms up to about a point (seed 67.4% there,
68.5% here) because Jev is sampled and each pass is its
own measurement. Quoting across the two would compare arms that never sat the
same exam — the reporter renders them as separate sections for that reason.

## Does the search beat doing something dumb?

The control to beat: skip the search entirely and attach two *random* sound
clauses per option, drawn from the same gated pool. Run as a distribution over
ten seeds rather than as ten arms, because reading the best or worst seed is a
forking path.

**The search wins: better on 8 of 10 seeds, worse on none, ties on 2. Sign test
p = 0.008.**

That is a real effect, but note how close the control gets: its mean is 81.1%
against the evolved arm's 87.6%, its best seed matches the evolved arm outright,
and pair by pair only 3 of the 10 separate at p < 0.05. Most of the work is
being done by the evidence gate and the clause grammar; the Pareto search on top
is worth about six points.

## The noise floor

The same prompt was entered twice as two arms, differing only by a trailing
space to defeat the text cache. Its self-disagreement is the floor below which
no comparison means anything.

The twins disagree on **1 of 89** instances, so comparisons here are about
prompts rather than sampling. Even so, **7 of the 15 arms do not separate from
that floor** — including the evolved arm itself and six of the ten random ones.
Across all 120 pairs only 65 separate at p < 0.05, and with that many tests
about 6 false positives are expected.

Without a floor arm a column of accuracies reads like a ranking. With one, it is
clear how much of the table is unorderable at this sample size.

## Is Jev's own judgement load-bearing?

The proposer asks Jev to pick which repair to apply. An ablation replaces that
with the statistically strongest shortlist entry.

| | with Jev choosing | argmax |
| --- | ---: | ---: |
| test accuracy | 87.6% | 80.9% |

**+6.7 points, p = 0.070 on 7/1 discordant instances — not significant.** It
leans in Jev's favour, but this data cannot establish it. An earlier version of
this write-up claimed it was established; that was wrong.

## A human still wins

The hand-written reference reaches 97.8% against the evolved 87.6%, on 9 vs 0
discordant instances, **p = 0.004**. That gap is real. The optimiser closes most
of the distance from a naive seed to a careful human, and does not close it.

## What the search produced

From a seed whose options said only what each action does mechanically:

```
auto_remediate:  Only choose this when the blast radius is small -- tier3, or
                 tier2 with a sev3 symptom. Never when service_tier is "tier1".
                 Never when known_runbook is "no".
rollback_deploy: Never choose this when recent_deploy is "no".
suppress:        Never choose this when duplicate_of_open_incident is "no".
page_oncall:     Never choose this when severity is "sev3".
```

Those are the real labelling rules, recovered from failure statistics and Jev's
own confusions, with no generative model in the loop. Cost: $0.17.

It also produced redundancy — `suppress` carries both a `never` clause and a
`prefer` clause encoding the same constraint, because equivalent clauses are
collapsed within one shortlist but not across iterations.

## Honest summary

It works: a naive prompt becomes a good one automatically, for cents, with no
text-generating model anywhere. It does not beat a careful human. It beats a
random draw from its own clause pool, but only just, and that pool exists
because of the evidence gate rather than the search.

The method's ceiling is its grammar. It searches a space of clauses the schema
and templates can express, which is why it cannot invent phrasing the way a
generative reflector would — and why a random draw from that same space is so
hard to beat.

## Reproducing

```sh
jevopt optimize  --task jevopt.tasks.triage --budget 6000
jevopt baselines --task jevopt.tasks.triage --k 2 --random-seeds 10
jevopt compare   --task jevopt.tasks.triage --candidate gepa=out/triage.results.json \
                 --greedy 2 --random-seeds 10
jevopt report    --task jevopt.tasks.triage "triage=out/triage.results.json" \
                 --compare "triage=out/triage.compare.json"
```

The recorded artifacts are no longer kept in the repo, so these figures cannot
be recomputed from it -- rerun the commands above to regenerate them. A copy of
the run the report tests pin lives in `tests/fixtures/`.
