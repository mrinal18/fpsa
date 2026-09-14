"""Label-blind ARC candidate ranking; separate task-macro scoring.

Candidate files contain inverse-transformed original-coordinate output grids.
Input rows: task_id, query_index, grid, source, candidate_id, optional confidence.
No targets enter rank_candidates. Oracle coverage is evaluation-only and is
never used for selecting the two submitted answers.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from .codec import grid_key


def rank_candidates(records):
    grouped = defaultdict(dict)
    seen = set()
    invalid = 0
    for row in records:
        task, query = str(row['task_id']), int(row['query_index'])
        uid = (task, query, str(row['source']), str(row['candidate_id']))
        if uid in seen:
            raise ValueError(f'Duplicate candidate event {uid}; refusing vote inflation')
        seen.add(uid)
        try:
            key = grid_key(row['grid'])
        except (ValueError, TypeError):
            invalid += 1
            continue
        q = float(row.get('confidence', 0.))
        if not math.isfinite(q):
            q = 0.
        item = grouped[(task, query)].setdefault(key, dict(grid=row['grid'], votes=0, confidence_sum=0.))
        item['votes'] += 1
        item['confidence_sum'] += q
    ranked = {}
    for pair, items in grouped.items():
        ranked[pair] = sorted(items.values(), key=lambda v: (
            -v['votes'], -v['confidence_sum']/v['votes'], grid_key(v['grid'])))
    return ranked, dict(candidate_events=len(seen), invalid_candidates=invalid)


def score_candidates(records, solutions):
    ranked, diagnostics = rank_candidates(records)  # label-blind first
    scores = {k: [] for k in ('top1', 'top2', 'oracle_coverage')}
    per_task = {}
    for task, outputs in solutions.items():
        if not outputs:
            raise ValueError(f'No test outputs for {task}')
        totals = dict.fromkeys(scores, 0.)
        for i, target in enumerate(outputs):
            candidates = ranked.get((task, i), [])
            target_key = grid_key(target)
            hits = [grid_key(v['grid']) == target_key for v in candidates]
            totals['top1'] += any(hits[:1])
            totals['top2'] += any(hits[:2])
            totals['oracle_coverage'] += any(hits)
        per_task[task] = {k: v/len(outputs) for k, v in totals.items()}
        for k in scores:
            scores[k].append(per_task[task][k])
    # Missing predictions score zero, and each original task has equal weight.
    result = {k: 100.*sum(v)/max(1, len(v)) for k, v in scores.items()}
    result.update(diagnostics, tasks=len(solutions), per_task=per_task)
    submission = {}
    for task, outputs in solutions.items():
        submission[task] = []
        for i in range(len(outputs)):
            top = [v['grid'] for v in ranked.get((task, i), [])[:2]]
            # Placeholders are NOT inserted into the scored candidate pool.
            top = top or [[[0]]]
            submission[task].append({'attempt_1': top[0], 'attempt_2': top[-1]})
    return result, submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', nargs='+', required=True)
    p.add_argument('--solutions', required=True, help='Scoring only; NEVER passed to model/ranker')
    p.add_argument('--output', required=True)
    p.add_argument('--canonical-only', action='store_true', help='Exclude transformed candidates')
    a = p.parse_args()
    records = [json.loads(line) for fn in a.predictions for line in Path(fn).read_text().splitlines() if line.strip()]
    if a.canonical_only:
        records = [r for r in records if r.get('canonical', False)]
    result, submission = score_candidates(records, json.loads(Path(a.solutions).read_text()))
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    dest.with_name(dest.stem + '_submission.json').write_text(json.dumps(submission))
    print(json.dumps({k: v for k, v in result.items() if k != 'per_task'}, indent=2))

if __name__ == '__main__':
    main()
