"""Offline integration test on a synthetic color-permutation task, NOT ARC score."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from src.fpsa_arc import ARCConfig, ARCReasoner
from src.fpsa_arc.losses import supervised_loss


def run(output='results/arc_smoke', steps=8, device='cpu'):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    cfg = ARCConfig(hidden_size=24, num_heads=4, layers=2, seq_len=9,
                    puzzle_emb_len=0, max_iter=40, max_iter_eval=40,
                    refinement_steps=2)
    model = ARCReasoner(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = {k: p.detach().clone() for k, p in model.named_parameters()}
    history = []
    # Fixed development batch: overfit signal only; not generalization evidence.
    x = torch.randint(2, 12, (4, 9), device=device)
    y = 2 + ((x - 2 + 1) % 10)
    for update in range(steps):
        carry = None
        for segment in range(cfg.refinement_steps):
            optimizer.zero_grad(set_to_none=True)
            carry, out = model.forward_segment(x, carry)
            loss, metrics = supervised_loss(out, y)
            loss.backward()
            dense = [p for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(p.grad).all() for p in dense)
            optimizer.step()
            with torch.no_grad():
                for k, p in model.named_parameters():
                    ema[k].lerp_(p, 1 - .999)
            history.append(dict(update=update, segment=segment,
                                loss=float(loss.detach()), **metrics,
                                nfe=out['nfe'], mlp_calls=out['mlp_calls'],
                                solver=[i.as_dict() for i in out['infos']]))
    result = dict(task='synthetic_color_shift_not_ARC', device=device,
                  cuda_tested=str(device).startswith('cuda'), config=asdict(cfg),
                  parameters=model.parameter_report(), history=history)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'summary.json').write_text(json.dumps(result, indent=2))
    torch.save(dict(model=model.state_dict(), ema=ema, config=asdict(cfg)), root / 'smoke.pt')
    print(json.dumps(dict(task=result['task'], updates=len(history),
                         first_loss=history[0]['loss'], final_loss=history[-1]['loss'],
                         max_forward_residual=max(s['max_residual'] for h in history for s in h['solver']),
                         max_backward_residual=max(s['backward_residual'] for h in history for s in h['solver'])), indent=2))
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/arc_smoke')
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    run(a.output, a.steps, a.device)
