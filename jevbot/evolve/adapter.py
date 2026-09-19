"""GEPA adapter: run a candidate prompt over instances, score it, trace it.

Two things here are deliberately not the GEPA default.

Score is a *margin*, not accuracy. Jev returns the whole distribution, so
`p(best acceptable) - p(best unacceptable)` is available for free. It is
positive exactly when the answer is right, but unlike 0/1 it keeps moving while
a candidate is still wrong, which gives the search something to climb.

Traces carry the distribution, not just the pick. That is what lets the proposer
do credit assignment without a language model: when two options split 0.33/0.32,
the numbers say which pair of descriptions failed to separate.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from gepa.core.adapter import EvaluationBatch

from .. import client
from ..sim import ACTIONS
from . import prompt

WORKERS = 16


def _score_one(instance: dict, questions: dict) -> dict:
    state = instance["state"]
    good = set(instance["acceptable"])
    try:
        body = client.ask(state, questions)
        answer = body["answers"]["action"]
    except Exception as exc:                      # never fail a whole batch
        return {"error": str(exc), "chosen": None, "correct": False,
                "margin": -1.0, "probabilities": {}, "confidence": 0.0,
                "state": state, "acceptable": sorted(good)}

    probs = {a: float(answer.get("probabilities", {}).get(a, 0.0)) for a in ACTIONS}
    best_good = max((probs[a] for a in good), default=0.0)
    best_bad = max((probs[a] for a in ACTIONS if a not in good), default=0.0)
    chosen = answer["choice"]
    return {
        "state": state, "acceptable": sorted(good), "chosen": chosen,
        "correct": chosen in good, "margin": best_good - best_bad,
        "probabilities": probs, "confidence": float(answer.get("confidence", 0.0)),
        "cost": float(body.get("usage", {}).get("cost", 0.0) or 0.0),
    }


class JevAdapter:
    """Implements the GEPAAdapter protocol for the Jev action policy."""

    # The engine reads this attribute directly rather than probing for it, so it
    # has to exist. None means "the custom proposer owns text proposal".
    propose_new_texts = None

    def __init__(self) -> None:
        self.calls = 0
        self.spend = 0.0

    def evaluate(self, batch, candidate, capture_traces: bool = False):
        questions = prompt.questions(candidate)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            traces = list(pool.map(lambda i: _score_one(i, questions), batch))

        self.calls += len(batch)
        self.spend += sum(t.get("cost", 0.0) for t in traces)
        # Map margin in [-1, 1] onto [0, 1]; > 0.5 exactly when the pick is right.
        scores = [(1.0 + t["margin"]) / 2.0 for t in traces]
        outputs = [{"chosen": t["chosen"], "correct": t["correct"]} for t in traces]
        return EvaluationBatch(
            outputs=outputs, scores=scores,
            trajectories=traces if capture_traces else None,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """Per component, the evidence the proposer needs to repair its text."""
        out: dict[str, list[dict]] = {}
        for component in components_to_update:
            action = component[len(prompt.PREFIX):]
            records = []
            for trace in eval_batch.trajectories or []:
                if trace.get("chosen") is None:
                    continue
                good, chosen = set(trace["acceptable"]), trace["chosen"]
                probs = trace["probabilities"]
                if chosen == action and action not in good:
                    role = "chosen_wrongly"
                elif action in good and chosen != action:
                    role = "missed"
                elif action in good and chosen == action:
                    role = "correct"
                else:
                    continue
                records.append({
                    "role": role,
                    "state": trace["state"],
                    "should_have_been": trace["acceptable"],
                    "model_chose": chosen,
                    "p_this_action": round(probs.get(action, 0.0), 3),
                    "p_model_chose": round(probs.get(chosen, 0.0), 3),
                    "confidence": round(trace["confidence"], 3),
                })
            out[component] = records
        return out
