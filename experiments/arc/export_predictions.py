"""Convert official TRM/FPSA eval_save_outputs to label-free candidate JSONL."""
import argparse
import json
from pathlib import Path
import torch
from .codec import decode, inverse, parse_identifier, grid_key


def export(path, identifiers, challenges, source):
    data = torch.load(path, map_location='cpu', weights_only=True)
    required = ('inputs', 'preds', 'puzzle_identifiers', 'q_halt_logits')
    if not all(k in data for k in required):
        raise ValueError(f'Enable eval_save_outputs={required} in the official run')
    rows = []
    lookup = {}
    for task, challenge in challenges.items():
        lookup[task] = defaultdict_indices(challenge['test'])
    for j, identifier in enumerate(data['puzzle_identifiers'].tolist()):
        if identifier == 0:  # upstream padded batch row
            continue
        name = identifiers[identifier]
        task, tid, colors = parse_identifier(name)
        inp = decode(data['inputs'][j].numpy())
        pred = decode(data['preds'][j].numpy())
        if inp is None:
            raise ValueError('Cannot decode model input')
        inp = inverse(inp, tid, colors)
        indices = lookup[task].get(grid_key(inp))
        if indices is None:
            raise ValueError(f'Input not found in challenge {task}')
        for qi in indices:
            rows.append(dict(task_id=task, query_index=qi,
                grid=None if pred is None else inverse(pred, tid, colors).tolist(),
                confidence=float(data['q_halt_logits'][j].double().sigmoid()),
                source=source, candidate_id=f'{name}:{j}', canonical='|||' not in name))
    return rows


def defaultdict_indices(pairs):
    result = {}
    for i, pair in enumerate(pairs):
        result.setdefault(grid_key(pair['input']), []).append(i)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--identifiers', required=True)
    p.add_argument('--challenges', required=True, help='Only test input grids are read')
    p.add_argument('--source', required=True, help='Unique checkpoint/rank ID')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows = export(a.predictions, json.loads(Path(a.identifiers).read_text()),
                  json.loads(Path(a.challenges).read_text()), a.source)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(v)+'\n' for v in rows))

if __name__ == '__main__':
    main()
