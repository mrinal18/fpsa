"""TRM-FP parity: coupled (z,y) fixed point, blended-3 values, stacked
implicit gradients. NO guard, NO jac-reg. Logs: joint residual, rounds,
coupled-rho audit, gate simplex means, disagreement ||LN(z)-LN(x)||."""
import argparse, csv, gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F, yaml
from data.parity import make_parity
from models.trmfp import TRMFPModel, coupled_audit
from utils.logging_utils import CSVLogger, set_seed


def evaluate(m, x, y, batch=512):
    m.eval(); ct = cs = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            lg, _ = m(x[i:i+batch]); p = lg.argmax(-1)
            ct += (p == y[i:i+batch]).sum().item()
            cs += (p == y[i:i+batch]).all(1).sum().item()
    m.train(); return ct / y.numel(), cs / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trmfp_parity.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_minutes", type=float, default=2.8)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); cfg["seed"] = a.seed
    torch.set_flush_denormal(True); set_seed(a.seed)
    rd = f"runs/{cfg['name']}_seed{a.seed}"; os.makedirs(rd, exist_ok=True)
    json.dump(cfg, open(f"{rd}/config.json", "w"), indent=2)
    ck, dn = f"{rd}/checkpoint.pt", f"{rd}/final.json"
    if os.path.exists(dn):
        print("complete:", open(dn).read()); return
    (xtr, ytr), (xte, yte) = make_parity(20000, 4000, 16, seed=1234)
    xs, ys = xte[:1024], yte[:1024]
    m = TRMFPModel(2, 2, d_model=cfg["d_model"], num_heads=cfg["num_heads"],
                   n_inner=cfg["n_inner"], T_outer=cfg["T_outer"], tol=cfg["tol"],
                   neumann_steps=cfg["neumann_steps"], max_seq_len=16)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    g = torch.Generator().manual_seed(a.seed); start = 0
    F_ = ["step","loss","tok","seq","joint_res","rounds","rho","bx","by","bz","disagree","sps"]
    if os.path.exists(ck):
        c = torch.load(ck, weights_only=False)
        m.load_state_dict(c["m"]); opt.load_state_dict(c["o"])
        g.set_state(c["g"]); torch.set_rng_state(c["r"]); start = c["s"]
        fh = open(f"{rd}/train_log.csv", "a", newline=""); w = csv.DictWriter(fh, fieldnames=F_)
        class L:
            def log(s, **k): w.writerow({x: k.get(x, "") for x in F_}); fh.flush()
            def close(s): fh.close()
        log = L(); print(f"resumed {start}")
    else:
        log = CSVLogger(f"{rd}/train_log.csv", F_)
        print(f"params {sum(p.numel() for p in m.parameters()):,}")
    def sv(s): torch.save({"m": m.state_dict(), "o": opt.state_dict(),
                           "g": g.get_state(), "r": torch.get_rng_state(), "s": s}, ck)
    m.train(); t0 = time.time(); step = start
    for step in range(start + 1, cfg["steps"] + 1):
        idx = torch.randint(0, len(xtr), (cfg["batch"],), generator=g)
        lg, rec = m(xtr[idx])
        loss = F.cross_entropy(lg.view(-1, 2), ytr[idx].view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        del lg
        if step % 50 == 0: gc.collect()
        if step % cfg["eval_every"] == 0 or step == 1:
            ta, sa = evaluate(m, xs, ys)
            est, rho = coupled_audit(m, xs[:64], iters=12)
            gm = rec["gate_mean"].tolist()
            sps = (time.time() - t0) / max(step - start, 1)
            log.log(step=step, loss=f"{loss.item():.4f}", tok=f"{ta:.4f}",
                    seq=f"{sa:.4f}", joint_res=f"{rec['joint_res']:.3e}",
                    rounds=rec["rounds"], rho=f"{rho:.3f}",
                    bx=f"{gm[0]:.3f}", by=f"{gm[1]:.3f}", bz=f"{gm[2]:.3f}",
                    disagree=f"{rec['disagree']:.3f}", sps=f"{sps:.2f}")
            print(f"step {step:4d} loss {loss.item():.4f} tok {ta:.4f} seq {sa:.4f} "
                  f"jres {rec['joint_res']:.2e} rho {rho:.3f} "
                  f"b=({gm[0]:.2f},{gm[1]:.2f},{gm[2]:.2f}) dis {rec['disagree']:.2f} ({sps:.2f}s)")
            sv(step)
            if ta >= 0.9999: break
        if (time.time() - t0) / 60 > a.max_minutes:
            sv(step); print(f"TIME LIMIT {step}"); log.close(); return
    ta, sa = evaluate(m, xte, yte)
    json.dump({"final_tok": ta, "final_seq": sa, "steps": step}, open(dn, "w"), indent=2)
    sv(step); print(f"FINAL tok {ta:.4f} seq {sa:.4f}"); log.close()


if __name__ == "__main__":
    main()
