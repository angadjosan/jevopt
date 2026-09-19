# Findings

Two tasks, each optimised from a deliberately naive seed. Held-out test set,
n=89, split by situation so nothing in test appears in train. Protocol fixed in
advance: [`preregistration.md`](preregistration.md).

## Headline

| | naive seed | **evolved** | hand-written | random clauses (10 seeds) |
| --- | ---: | ---: | ---: | ---: |
| alert triage | 68.5% | **87.6%** | 97.8% | 81.1% (71.9–87.6) |
| robot arm | 52.8% | **78.7%** | 79.8% | 82.2% (75.3–88.8) |

Every figure here is from the paired pass in `runs/*.compare.json`, where all
arms answered the same instances in one run. The per-run files report the same
arms up to about a point (triage seed 67.4% there, 68.5% here; robot evolved
80.9% there, 78.7% here) because Jev is sampled and each pass is its own
measurement. Quoting across the two would be comparing arms that never sat the
same exam -- the reporter renders them as separate sections for that reason.

Everything beats its naive seed by 20–26 points, and that is the one result
that is unambiguous.

## The pre-registered test

> GEPA-evolved vs the random-clause control, paired McNemar, two-sided, the
> control read as a *distribution* over 10 seeds rather than as 10 arms.

- **Triage: the search wins.** Better on 8 of 10 seeds, worse on none, ties on 2.
  Sign test **p = 0.008**.
- **Robot: it does not.** Better on 2 of 10, worse on 7. **p = 0.18.** Random is
  nominally ahead.

So attaching two *random* sound clauses per option is competitive with the whole
Pareto search, and on one of the two tasks it is not beaten. The evidence gate
and the clause grammar are doing much of the work.

## What the noise floor exposed

The same prompt was entered twice as two arms, differing only by a trailing
space to defeat the text cache. Its self-disagreement is the floor below which
no comparison means anything.

- **Triage**: the twins disagree on **1 of 89** instances. Comparisons there are
  about prompts.
- **Robot**: they disagree on **6 of 89** — and **13 of the 14 arms are not
  distinguishable from that floor.** On the robot task, almost none of the
  differences in the headline table are interpretable at this sample size.

That is the most useful thing the controls bought. Without the floor arm, the
robot column reads like a ranking; with it, the column is mostly noise.

## Is Jev's own judgement load-bearing?

The proposer asks Jev to pick which repair to apply. An ablation replaces that
with the statistically strongest shortlist entry.

| | with Jev choosing | argmax |
| --- | ---: | ---: |
| triage | 87.6% | 80.9% |

**+6.7 points, p = 0.070 on 7/1 discordant instances — not significant.** It
leans in Jev's favour on both tasks, but this data cannot establish it. An
earlier version of this write-up claimed it was established; that was wrong.

## Auto-derived conditions beat hand-written ones

Same robot task, same algorithm, only the condition vocabulary changed:

| vocabulary | test |
| --- | ---: |
| 21 conditions written by hand | 62.6% |
| 29 derived mechanically from the state schema + 4 hand-written groupings | **80.9%** |

Enumerating `field is "value"` off the states outperformed the vocabulary
designed for the task. Hand-writing conditions is not where the effort pays.

## What the search actually produced

Triage, from a seed that said only what each action does mechanically:

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

It works: naive prompts become good ones automatically, for cents, with no
text-generating model. It does not beat a careful human on triage (97.8% vs
87.6%, and that gap **is** significant at p=0.004). On the robot it ties the
human — but on the robot almost nothing separates from noise, so that tie is
not much of a claim.

The method's ceiling is its grammar. It searches a space of clauses the schema
and templates can express, which is why it cannot invent the phrasing a
generative reflector would, and why a random draw from that same space is so
hard to beat.

## Reproducing

```sh
jevopt optimize  --task jevopt.tasks.triage --budget 6000
jevopt baselines --task jevopt.tasks.triage --k 2 --random-seeds 10
jevopt compare   --task jevopt.tasks.triage --candidate gepa=runs/triage.results.json \
                 --greedy 2 --random-seeds 10
jevopt report    --task jevopt.tasks.triage "triage=runs/triage.results.json" \
                 --compare "triage=runs/triage.compare.json"
```

Recorded artifacts are in [`../runs/`](../runs), including per-instance
correctness vectors so the comparisons can be re-analysed without spending
anything.
