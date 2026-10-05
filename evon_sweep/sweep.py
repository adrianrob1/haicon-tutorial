"""Sweep 8000 EVON hyperparameter configurations on the 2D logistic regression
problem from notebook 1 and dump everything the explorer UI needs to JSON.

    uv run python evon_sweep/sweep.py            # trains missing runs, writes evon_sweep/sweep.json
    uv run python evon_sweep/sweep.py --analyze  # (re)compute references/metrics only; resumes from sweep.json.partial
    uv run python evon_sweep/build_ui.py         # writes evon_sweep/explorer.html
"""

import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # CPU only; torch.compile in forked workers must not touch CUDA

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Setup from 1_logistic2d.ipynb
# ---------------------------------------------------------------------------

N = 50
DELTA = 0.01  # prior precision
STEPS = 10_000
W0 = (3.0, 2.0)


def toydata(N=50):
    y = torch.cat([torch.zeros(N // 2), torch.ones(N - N // 2)])
    c = torch.tensor([[-5.0, 1.0], [1.0, 5.0]])  # centers of the blobs
    s = torch.tensor([1.2, 1.0])  # std of the blobs
    X = c[y.long()] + s[y.long(), None] * torch.randn(N, 2)
    return X, y


def get_data():
    torch.manual_seed(2)
    return toydata(N=N)


def total_loss(W, X, y):
    """Negative log joint (up to a constant) for a batch of weights W: (M, 2)."""
    nll = F.binary_cross_entropy_with_logits(W @ X.T, y[None].expand(W.shape[0], -1), reduction='none').sum(1)
    return nll + 0.5 * DELTA * W.square().sum(1)


# ---------------------------------------------------------------------------
# Hyperparameter grid: 2 x 5 x 5 x 5 x 2 x 2 x 2 x 4 = 8000 configurations
# ---------------------------------------------------------------------------

GRID = {
    'whiten_prec_grad': [True, False],
    'lr': [0.03, 0.1, 0.3, 1.0, 3.0],
    'hess_init': [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    'beta2': [0.99, 0.999, 0.9995, 0.9999, 0.99999],
    'beta1': [0.7, 0.9],
    'ess_mult': [1, 10],  # ess = ess_mult * N; 1 targets the Bayesian posterior, 10 a colder one
    'phasing': [False, True],  # alternate clean (precond) / noisy (Hessian) steps
    'beta3': [-1, 0.9, 0.99, 0.999],  # EVON's shampoo_beta (Kronecker-factor EMA); -1 = tied to beta2
}
# the first sweep had only these axes; a run's noise seed depends on them alone, so
# every (phasing, beta3) slice sees the same random draws (and the original 1000
# runs are exactly the phasing=False, beta3=-1 slice)
BASE_KEYS = ['whiten_prec_grad', 'lr', 'hess_init', 'beta2', 'beta1', 'ess_mult']


def base_index(cfg):
    i = 0
    for k in BASE_KEYS:
        i = i * len(GRID[k]) + GRID[k].index(cfg[k])
    return i

# log-ish spaced checkpoints so both the early transient and the end are resolved
CKPTS = sorted(set(np.unique(np.round(np.geomspace(1, STEPS, 60))).astype(int).tolist()) | set(range(0, STEPS + 1, 125)))


def configs():
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]


def evon_cov(opt, p, ess, wd):
    """Exact covariance of EVON's Gaussian: noise ~ N(0, 1/(ess (h + wd))) in the
    Shampoo eigenbasis, projected back with Q. For a (1, 2) weight this is
    Q2 diag(var) Q2^T (Q1 = +-1)."""
    st = opt.state[p]
    var = 1.0 / (ess * (st['h_mom'].reshape(-1).double() + wd))
    if 'Q' not in st:
        return torch.diag(var)
    Q2 = st['Q'][1].double()
    return Q2 @ torch.diag(var) @ Q2.T


def run(args):
    idx, cfg = args
    torch.set_num_threads(1)
    from evon import EVON

    X, y = get_data()
    torch.manual_seed(1000 + base_index(cfg))

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight[:] = torch.tensor(W0)
    ess = cfg['ess_mult'] * N
    wd = DELTA / N
    opt = EVON(
        model.parameters(),
        lr=cfg['lr'],
        ess=ess,
        weight_decay=wd,
        hess_init=cfg['hess_init'],
        betas=(cfg['beta1'], cfg['beta2']),
        shampoo_beta=cfg['beta3'],
        phasing=cfg['phasing'],
        mc_samples=1,
        precondition_frequency=1,
        whiten_prec_grad=cfg['whiten_prec_grad'],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=0.0, T_max=STEPS)
    p = model.weight

    ck = set(CKPTS)
    traj = []  # (step, mu1, mu2, s11, s12, s22)

    def record(step):
        with torch.no_grad():
            mu = p.detach().reshape(-1).double()
            S = evon_cov(opt, p, ess, wd)
        traj.append([step, mu[0].item(), mu[1].item(), S[0, 0].item(), S[0, 1].item(), S[1, 1].item()])

    t0 = time.time()
    diverged = False
    for step in range(1, STEPS + 1):
        with opt.sampled_params(train=True):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(X).squeeze(-1), y)
            loss.backward()
        opt.step()
        sched.step()
        if step == 1:
            traj.insert(0, [0, *W0, 1 / (ess * (cfg['hess_init'] + wd)), 0.0, 1 / (ess * (cfg['hess_init'] + wd))])
        if step in ck:
            record(step)
        if not torch.isfinite(p).all() or p.abs().max() > 1e4:
            diverged = True
            break

    out = dict(cfg)
    out['idx'] = idx
    out['ess'] = ess
    out['diverged'] = diverged
    out['traj'] = traj
    out['seconds'] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Reference posteriors: exact (grid quadrature) and Laplace at the MAP
# ---------------------------------------------------------------------------

def exact_refs():
    X, y = get_data()
    X, y = X.double(), y.double()
    lo, hi, n = -15.0, 60.0, 751
    g = torch.linspace(lo, hi, n, dtype=torch.float64)
    g1, g2 = torch.meshgrid(g, g, indexing='xy')
    W = torch.stack([g1.ravel(), g2.ravel()], 1)
    L = total_loss(W, X, y)
    dA = (g[1] - g[0]) ** 2

    def moments(temp):  # posterior with likelihood^temp prior^temp (EVON with ess = temp N)
        logp = -temp * L
        c = logp.max()
        p = (logp - c).exp()
        Z = p.sum() * dA
        p = p / Z
        mu = (p[:, None] * W).sum(0) * dA
        D = W - mu
        S = (p[:, None, None] * D[:, :, None] * D[:, None, :]).sum(0) * dA
        edge = p.reshape(n, n)
        edge_mass = (edge[0].sum() + edge[-1].sum() + edge[:, 0].sum() + edge[:, -1].sum()) * dA
        logZ = torch.log(Z) + c  # log normalizer of exp(-temp L)
        return mu, S, logZ.item(), edge_mass.item()

    exact = {}
    for m in GRID['ess_mult']:
        mu, S, logZ, edge = moments(float(m))
        exact[m] = dict(mu=mu.tolist(), cov=S.tolist(), logZ=logZ, edge_mass=edge)
        print(f'exact posterior ess={m}N: mu={mu.tolist()} edge mass={edge:.2e}', file=sys.stderr)
    return exact


def laplace_ref():
    """MAP by Newton; Laplace precision = Hessian (scaled by ess/N). Uses autograd,
    so call it only after the forked worker pools are done."""
    X, y = get_data()
    X, y = X.double(), y.double()
    w = torch.tensor(W0, dtype=torch.float64)
    f = lambda w: total_loss(w[None], X, y)[0]
    for _ in range(100):
        gr = torch.func.grad(f)(w)
        H = torch.func.hessian(f)(w)
        w = w - torch.linalg.solve(H, gr)
    H = torch.func.hessian(f)(w)
    return dict(mu=w.tolist(), cov={m: (torch.linalg.inv(H) / m).tolist() for m in GRID['ess_mult']})


def kl_to_exact(mu_q, S_q, temp, X, y, logZ, nmc=16000, seed=0):
    """KL(q || p_temp) = E_q[log q] + temp E_q[L] + log Z_temp, MC over q with
    common random numbers across configs. Batched over leading dims of mu_q/S_q."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(nmc, 2, generator=gen, dtype=torch.float64)
    Lc = torch.linalg.cholesky(S_q)
    Wq = mu_q[..., None, :] + z @ Lc.transpose(-1, -2)
    ent = 0.5 * torch.logdet(2 * math.pi * math.e * S_q)
    EL = total_loss(Wq.reshape(-1, 2), X, y).reshape(Wq.shape[:-1]).mean(-1)
    return -ent + temp * EL + logZ


def gauss_kl(mu0, S0, mu1, S1):
    """KL(N0 || N1)."""
    S1i = torch.linalg.inv(S1)
    d = mu1 - mu0
    return 0.5 * (torch.trace(S1i @ S0) + d @ S1i @ d - 2 + torch.logdet(S1) - torch.logdet(S0)).item()


def best_gaussian(temp, X, y, logZ, mu0, S0):
    """argmin_q KL(q || p_temp) over full-covariance Gaussians: the floor no
    Gaussian method (EVON included) can beat. Reparameterised, fixed draws, LBFGS."""
    gen = torch.Generator().manual_seed(123)
    z = torch.randn(32000, 2, generator=gen, dtype=torch.float64)
    mu = torch.tensor(mu0, dtype=torch.float64, requires_grad=True)
    C = torch.linalg.cholesky(torch.tensor(S0, dtype=torch.float64))
    ld = C.diagonal().log().clone().requires_grad_(True)
    off = C[1, 0].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([mu, ld, off], max_iter=500, line_search_fn='strong_wolfe')

    def chol():
        return torch.stack([torch.stack([ld[0].exp(), torch.zeros((), dtype=torch.float64)]), torch.stack([off, ld[1].exp()])])

    def closure():
        opt.zero_grad()
        Lc = chol()
        f = -ld.sum() + temp * total_loss(mu + z @ Lc.T, X, y).mean()
        f.backward()
        return f

    for _ in range(3):
        opt.step(closure)
    with torch.no_grad():
        Lc = chol()
        S = Lc @ Lc.T
        kl = kl_to_exact(mu, S, temp, X, y, logZ).item()
    return dict(mu=mu.detach().tolist(), cov=S.tolist(), kl=kl)


def run_ivon(ess_mult):
    """IVON with notebook 1's settings, recorded at the same checkpoints."""
    from ivon import IVON

    torch.set_num_threads(1)
    X, y = get_data()
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight[:] = torch.tensor(W0)
    ess, wd = ess_mult * N, DELTA / N
    opt = IVON(model.parameters(), lr=5.0, ess=ess, weight_decay=wd, hess_init=0.005, beta1=0.9, beta2=0.9995, rescale_lr=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=0.0, T_max=STEPS)
    var0 = 1 / (ess * (0.005 + wd))
    traj = [[0, *W0, var0, 0.0, var0]]
    ck = set(CKPTS)
    for step in range(1, STEPS + 1):
        with opt.sampled_params(train=True):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(X).squeeze(-1), y).backward()
        opt.step()
        sched.step()
        if step in ck:
            mu = model.weight.detach().reshape(-1).double()
            var = 1 / (ess * (opt.param_groups[0]['hess'].double() + wd))
            traj.append([step, mu[0].item(), mu[1].item(), var[0].item(), 0.0, var[1].item()])
    return traj


def traj_kls(traj, temp, X, y, logZ):
    """KL(q_t || exact) along a trajectory (None where the run had blown up)."""
    T = torch.tensor([t[1:] for t in traj], dtype=torch.float64)
    ok = torch.isfinite(T).all(1)
    out = [None] * len(traj)
    if ok.any():
        Tk = T[ok]
        mu = Tk[:, :2]
        S = torch.stack([torch.stack([Tk[:, 2], Tk[:, 3]], -1), torch.stack([Tk[:, 3], Tk[:, 4]], -1)], -2)
        kls = torch.cat([kl_to_exact(mu[i:i + 32], S[i:i + 32], temp, X, y, logZ) for i in range(0, len(mu), 32)])
        for i, v in zip(ok.nonzero().reshape(-1).tolist(), kls.tolist()):
            out[i] = v if math.isfinite(v) else None
    return out


def cfg_key(r):
    # runs from before phasing/beta3 existed are the phasing=False, beta3=-1 slice
    return tuple(r.get(k, GRID[k][0]) for k in GRID)


def train_all(previous):
    """Train every config not already in `previous` (same config => same seed => same run)."""
    cfgs = configs()
    assert len(cfgs) == 8000, len(cfgs)
    done = {cfg_key(r): r for r in previous}
    results = [None] * len(cfgs)
    todo = []
    for idx, cfg in enumerate(cfgs):
        r = done.get(cfg_key(cfg))
        if r is None:
            todo.append((idx, cfg))
        else:
            results[idx] = {**r, **cfg, 'idx': idx}
    workers = int(os.environ.get('WORKERS', 48))
    print(f'{len(cfgs)} configs, {len(cfgs) - len(todo)} reused, {len(todo)} to train on {workers} workers, '
          f'{len(CKPTS)} checkpoints', file=sys.stderr)
    t0 = time.time()
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run, todo, chunksize=4)):
            results[r['idx']] = r
            if k % 250 == 0:
                print(f'[{k}/{len(todo)}] {time.time() - t0:.0f}s', file=sys.stderr)
    return results


_EXACT = None


def run_metrics(r):
    """Per-run KL trajectory and final metrics (gap to the floor is filled in later)."""
    torch.set_num_threads(1)
    X, y = get_data()
    Xd, yd = X.double(), y.double()
    m = r['ess_mult']
    ex = _EXACT[m]
    _, m1, m2, s11, s12, s22 = r['traj'][-1]
    out = dict(idx=r['idx'], klt=traj_kls(r['traj'], float(m), Xd, yd, ex['logZ']))
    if (not r['diverged']) and all(map(math.isfinite, [m1, m2, s11, s12, s22])):
        mu = torch.tensor([m1, m2], dtype=torch.float64)
        S = torch.tensor([[s11, s12], [s12, s22]], dtype=torch.float64)
        out['kl_exact'] = kl_to_exact(mu, S, float(m), Xd, yd, ex['logZ']).item()
        out['kl_gauss'] = gauss_kl(mu, S, torch.tensor(ex['mu'], dtype=torch.float64), torch.tensor(ex['cov'], dtype=torch.float64))
        out['loss'] = total_loss(mu[None], Xd, yd)[0].item()
        out['corr'] = s12 / math.sqrt(s11 * s22)
    else:
        out['kl_exact'] = out['kl_gauss'] = out['loss'] = out['corr'] = None
    return out


def analyze(results, recompute=False):
    """References (exact, best Gaussian, Laplace, IVON) and per-run metrics. Runs that
    already carry metrics (from an earlier analysis) are skipped unless `recompute`."""
    global _EXACT
    exact = _EXACT = exact_refs()
    X, y = get_data()
    Xd, yd = X.double(), y.double()

    todo = [r for r in results if recompute or 'klt' not in r or 'kl_exact' not in r]
    t0 = time.time()
    workers = int(os.environ.get('WORKERS', 48))
    print(f'metrics for {len(todo)} runs on {workers} workers', file=sys.stderr)
    with Pool(workers) as pool:
        for k, out in enumerate(pool.imap_unordered(run_metrics, todo, chunksize=8)):
            results[out['idx']].update(out)
            if k % 500 == 0:
                print(f'[{k}/{len(todo)}] {time.time() - t0:.0f}s', file=sys.stderr)
    print(f'per-run metrics in {time.time() - t0:.0f}s', file=sys.stderr)

    torch.set_num_threads(min(32, os.cpu_count()))
    laplace = laplace_ref()
    ivon = {}
    for m in GRID['ess_mult']:
        traj = run_ivon(m)
        torch.set_num_threads(min(32, os.cpu_count()))
        ex = exact[m]
        # multi-start: moment-matched exact, shrunk Laplace, and the 5 best EVON finals
        starts = [(ex['mu'], ex['cov']), (laplace['mu'], (torch.tensor(laplace['cov'][m]) / 4).tolist())]
        finals = sorted((r for r in results if r['ess_mult'] == m and r.get('kl_exact') is not None), key=lambda r: r['kl_exact'])
        for r in finals[:5]:
            _, a, b, s11, s12, s22 = r['traj'][-1]
            starts.append(([a, b], [[s11, s12], [s12, s22]]))
        ex['qstar'] = min((best_gaussian(float(m), Xd, yd, ex['logZ'], mu0, S0) for mu0, S0 in starts), key=lambda q: q['kl'])
        lap_S = torch.tensor(laplace['cov'][m], dtype=torch.float64)
        ex['kl_laplace'] = kl_to_exact(torch.tensor(laplace['mu'], dtype=torch.float64), lap_S, float(m), Xd, yd, ex['logZ']).item()
        ivon[m] = dict(traj=traj, klt=traj_kls(traj, float(m), Xd, yd, ex['logZ']))
        print(f'ess={m}N: best-Gaussian KL floor {ex["qstar"]["kl"]:.3f}, Laplace {ex["kl_laplace"]:.3f}, '
              f'IVON {ivon[m]["klt"][-1]:.3f}', file=sys.stderr)

    for r in results:
        r['kl_gap'] = None if r['kl_exact'] is None else r['kl_exact'] - exact[r['ess_mult']]['qstar']['kl']

    return dict(
        setup=dict(N=N, delta=DELTA, steps=STEPS, w0=W0, X=X.tolist(), y=y.tolist()),
        grid=GRID,
        exact={str(k): v for k, v in exact.items()},
        laplace=dict(mu=laplace['mu'], cov={str(k): v for k, v in laplace['cov'].items()}),
        ivon={str(k): v for k, v in ivon.items()},
        runs=results,
    )


def main():
    """`sweep.py` trains and analyzes; `sweep.py --analyze` recomputes the
    references and metrics from the runs already stored in sweep.json."""
    path = os.path.join(HERE, 'sweep.json')
    partial = path + '.partial'  # trained runs whose analysis has not finished yet
    if '--analyze' in sys.argv:
        with open(partial if os.path.exists(partial) else path) as f:
            results = json.load(f)['runs']
    else:
        previous = []
        if os.path.exists(path):
            with open(path) as f:
                previous = json.load(f)['runs']
        results = train_all(previous)
        with open(partial, 'w') as f:  # keep the training if analysis fails; resume with --analyze
            json.dump(dict(runs=results), f)
    out = analyze(results, recompute='--recompute' in sys.argv)
    with open(path, 'w') as f:
        json.dump(out, f)
    if os.path.exists(partial):
        os.remove(partial)
    print(f'wrote {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
