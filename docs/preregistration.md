# Pre-registration: does the search earn its keep?

> **Note added later.** The second task referred to throughout ("the robot") has
> since been removed from this repo. This document is left exactly as written,
> because a pre-registration edited after seeing the results is not one.

Written before looking at any triage result. The robot numbers are already seen,
so everything below about the robot is exploratory by construction; triage is
the confirmatory test.

## The question

A Jev-native GEPA proposer evolves the option descriptions of a `choice`
question. On the robot task a search-free control — attach two randomly chosen
*sound* clauses per option — scored 67.4% against the optimiser's 62.6%. If that
holds up, the evidence gate and the clause grammar are doing the work and the
Pareto search is not earning its keep.

I have already seen that comparison, so it cannot also confirm it.

## Confirmatory test (triage, unseen at time of writing)

**One** pre-registered comparison:

> GEPA-evolved vs the random-clause control, paired McNemar on the same test
> instances, two-sided, alpha = 0.05.

- Direction is not predicted. Either sign is a result; "random wins" is the
  outcome I currently expect.
- The random control is a *distribution*, not an arm. Ten seeds; the comparison
  is against their mean accuracy, and the seed spread is reported next to it. A
  single lucky seed beating the optimiser is not evidence, and neither is a
  single unlucky one losing to it.
- `b + c` (the discordant count) is reported with the p-value. Below b+c = 6 no
  result can reach p < 0.05 whatever the split, so "not distinguishable" there
  is a statement about the design, not about the methods.

## Noise floor, measured not assumed

Jev is sampled, so some discordance between any two arms is response noise
rather than prompt difference. The same prompt is therefore entered twice as two
arms (forced past the text cache). Its self-discordance is the floor: no pair
whose b+c sits at or under that floor is interpretable at all.

## Everything else is exploratory

Every other pair — seed, hand-written reference, greedy, the ablation that
strips Jev from the repair decision — is reported without multiplicity
correction and labelled exploratory. With ~15 pairs, roughly one spurious
p < 0.05 is expected.

## Committed in advance

- Test instances are fixed by `task.split(seed=0)` and are not re-drawn.
- Per-instance correctness vectors are stored, so re-analysis costs no API calls
  and there is no incentive to re-run and re-pick.
- If the confirmatory test comes out "not distinguishable", that is the finding.
  It will not be replaced by a search over other comparisons that happen to be
  significant.
