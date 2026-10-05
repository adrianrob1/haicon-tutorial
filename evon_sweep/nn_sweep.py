"""Sweep 5184 EVON configurations on notebook 2's two-moons MLP and store what the
neural-network explorer needs: function-space predictions, per-layer posterior
covariance and curvature spectra, 1D loss slices, and training curves.

Runs are written one file per run (nn_runs/<idx>.json.gz), so the sweep can be
stopped and resumed; finished runs are skipped.

    uv run python evon_sweep/nn_sweep.py            # train missing runs (+ references)
    uv run python evon_sweep/nn_build.py            # pack into evon_sweep/site/nn.html
"""

import gzip
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # CPU only; torch.compile in forked workers must not touch CUDA

import contourpy
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'nn_runs')

# ---------------------------------------------------------------------------
# Setup from 2_classification.ipynb
# ---------------------------------------------------------------------------

STEPS = 7500
HIDDEN = 128


def train_data():
    # notebook 2 calls torch.manual_seed(2) before make_moons, but make_moons draws from
    # NumPy's global RNG, so the notebook's data changes on every kernel restart. Fix it here.
    X, y = make_moons(n_samples=200, noise=0.08, random_state=2)
    return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32).view(-1, 1)


def test_data(n=1000):
    X, y = make_moons(n_samples=n, noise=0.08, random_state=123)
    return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32).view(-1, 1)


def far_data(n=600):
    """A ring well outside the moons, for 'does the model know it does not know'."""
    g = torch.Generator().manual_seed(0)
    ang = torch.rand(n, generator=g) * 2 * math.pi
    rad = 4 + 2 * torch.rand(n, generator=g)
    return torch.stack([0.5 + rad * torch.cos(ang), 0.25 + rad * torch.sin(ang)], 1)


class MLP(torch.nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        layers = [torch.nn.Linear(2, hidden), torch.nn.ReLU()]
        for _ in range(3):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hidden, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# feature-space grid (covers notebook 2's view (-2, 3) x (-1.5, 2) with room around it)
GX, GY = (-3.0, 4.0), (-2.5, 3.0)
FINE, COARSE = (64, 50), (48, 38)


def grid(nx, ny):
    xs = torch.linspace(GX[0], GX[1], nx)
    ys = torch.linspace(GY[0], GY[1], ny)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # row j = y, col i = x
    return xs, ys, torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)


# ---------------------------------------------------------------------------
# Hyperparameter grid: 4 x 3 x 3 x 3 x 2 x 2 x 2 x 3 x 2 = 5184 configurations
# ---------------------------------------------------------------------------

GRID = {
    'whiten_prec_grad': [True, False],
    'lr_mult': [0.3, 1.0, 3.0],  # lr = lr_mult * base; base 1e-2 whitened (step length), 1e-3 not
    'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3],
    'hess_init': [1e-3, 4e-3, 1.6e-2],
    'ess': [1e4, 3e4, 1e5],
    'beta2': [0.999, 0.9995, 0.9999],
    'beta1': [0.7, 0.9],
    'precondition_frequency': [1, 10],
    'beta3': [-1, 0.99],  # shampoo_beta; -1 = tied to beta2
}
BASE_LR = {True: 1e-2, False: 1e-3}
# notebook 2's EVON cell is one grid point
NB = dict(whiten_prec_grad=False, lr_mult=1.0, weight_decay=1e-5, hess_init=4e-3, ess=3e4, beta2=0.9995,
          beta1=0.9, precondition_frequency=10, beta3=-1)

CKPTS = sorted(set(np.unique(np.round(np.geomspace(1, STEPS, 24))).astype(int).tolist()) | set(range(0, STEPS + 1, 750)))


def configs():
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]


def weight_matrices(model):
    return [p for p in model.parameters() if p.ndim == 2]


def strided_entries(p, n):
    """n entries of a weight matrix, strided over its flattened index (as in notebook 2)."""
    out_f, in_f = p.shape
    flat = torch.linspace(0, out_f * in_f - 1, steps=min(n, out_f * in_f)).round().long()
    return flat // in_f, flat % in_f


# ---------------------------------------------------------------------------
# Posterior structure helpers (EVON and IVON)
# ---------------------------------------------------------------------------

def evon_layer(opt, p, ess, wd, n=40):
    """Covariance of n strided entries, eigenbasis curvature spectrum, and the two
    extreme eigendirections of one weight matrix."""
    st = opt.state[p]
    h = st['h_mom'].double()
    s2 = 1.0 / (ess * (h + wd))
    Q0, Q1 = st['Q'][0].double(), st['Q'][1].double()
    aa, bb = strided_entries(p, n)
    T = Q0[aa].unsqueeze(2) * Q1[bb].unsqueeze(1)  # Cov(W_ab, W_cd) = sum_kl Q0[a,k]Q0[c,k] s2[k,l] Q1[b,l]Q1[d,l]
    cov = torch.einsum('kl,akl,bkl->ab', s2, T, T)
    return dict(cov=cov, h=h.flatten(), s2=s2, Q0=Q0, Q1=Q1)


def direction(layer, k, l, shape):
    """Weight-space direction of eigenbasis entry (k, l): outer(Q0[:, k], Q1[:, l])."""
    return torch.outer(layer['Q0'][:, k], layer['Q1'][:, l]).float().reshape(shape)


def spectrum(h, nq=48):
    hs = torch.sort(h.double()).values
    qs = torch.linspace(0, 1, nq, dtype=torch.float64)
    idx = (qs * (len(hs) - 1)).round().long()
    return hs[idx].tolist()


# ---------------------------------------------------------------------------
# Function-space evaluation
# ---------------------------------------------------------------------------

def entropy(p):
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * p.log() + (1 - p) * (1 - p).log())


@torch.no_grad()
def sample_logits(model, opt, Xs, n):
    """Logits of n posterior samples on each input set in Xs (list) -> list of (n, len(X))."""
    outs = [[] for _ in Xs]
    for _ in range(n):
        with opt.sampled_params(train=False):
            for o, X in zip(outs, Xs):
                o.append(model(X).squeeze(-1))
    return [torch.stack(o) for o in outs]


def bin_ece(p, y, bins=10):
    conf = torch.where(p >= .5, p, 1 - p)
    pred = (p >= .5).float()
    ece = 0.0
    for b in range(bins):
        m = (conf > .5 + .5 * b / bins) & (conf <= .5 + .5 * (b + 1) / bins)
        if m.any():
            ece += m.float().mean().item() * abs((pred[m] == y[m]).float().mean().item() - conf[m].mean().item())
    return ece


def auroc(neg, pos):
    """P(score_pos > score_neg): how well MI separates far-from-data points from test points."""
    s = torch.cat([neg, pos])
    r = torch.empty_like(s)
    r[s.argsort()] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    rp = r[len(neg):].sum()
    return ((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


def boundaries(logits_grid, xs, ys, max_pts=90):
    """Zero-level contour lines of each sampled logit grid, decimated, in data coordinates."""
    out = []
    for z in logits_grid:
        gen = contourpy.contour_generator(xs.numpy(), ys.numpy(), z.numpy().astype(np.float64))
        lines = []
        for ln in gen.lines(0.0):
            if len(ln) < 2:
                continue
            step = max(1, math.ceil(len(ln) / max_pts))
            sel = ln[::step]
            if not np.allclose(sel[-1], ln[-1]):
                sel = np.vstack([sel, ln[-1]])
            lines.append(np.round(sel, 3).tolist())
        out.append(lines)
    return out


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def build_opt(model, cfg):
    from evon import EVON
    return EVON(
        model.parameters(),
        lr=cfg['lr_mult'] * BASE_LR[cfg['whiten_prec_grad']],
        ess=cfg['ess'],
        weight_decay=cfg['weight_decay'],
        hess_init=cfg['hess_init'],
        betas=(cfg['beta1'], cfg['beta2']),
        shampoo_beta=cfg['beta3'],
        mc_samples=1,
        precondition_frequency=cfg['precondition_frequency'],
        whiten_prec_grad=cfg['whiten_prec_grad'],
    )


def finite_state(opt, wd):
    for st in opt.state.values():
        if 'h_mom' in st and (not torch.isfinite(st['h_mom']).all() or (st['h_mom'] + wd <= 0).any()):
            return False
    return True


def evaluate(model, opt, cfg, layer_fn, kind):
    """Everything the explorer shows for the final posterior."""
    X, y = train_data()
    Xt, yt = test_data()
    Xf = far_data()
    xs, ys, G = grid(*FINE)
    S = 48
    lt, lf, lg, lx = sample_logits(model, opt, [Xt, Xf, G, X], S)
    pt, pf, pg = torch.sigmoid(lt), torch.sigmoid(lf), torch.sigmoid(lg)
    mi = lambda P: entropy(P.mean(0)) - entropy(P).mean(0)
    pbar_t = pt.mean(0).clamp(1e-6, 1 - 1e-6)
    ytf = yt.squeeze(-1)
    with torch.no_grad():
        mean_logit_g = model(G).squeeze(-1)
        nll_mean_train = F.binary_cross_entropy_with_logits(model(X), y).item()
    nll_samples_train = F.binary_cross_entropy_with_logits(lx, y.squeeze(-1).expand_as(lx), reduction='none').mean(1)
    mi_t, mi_f = mi(pt), mi(pf)
    out = dict(
        test_nll=-(ytf * pbar_t.log() + (1 - ytf) * (1 - pbar_t).log()).mean().item(),
        test_acc=((pbar_t > .5).float() == ytf).float().mean().item(),
        ece=bin_ece(pbar_t, ytf),
        mi_far=mi_f.mean().item(),
        mi_test=mi_t.mean().item(),
        auroc=auroc(mi_t, mi_f),
        train_nll_mean=nll_mean_train,
        sample_nll=sorted(nll_samples_train.tolist()),
        pbar=pg.mean(0).tolist(),
        mi_grid=mi(pg).tolist(),
        boundary_mean=boundaries(mean_logit_g.view(1, FINE[1], FINE[0]), xs, ys)[0],
        boundary_samples=boundaries(lg[:16].view(16, FINE[1], FINE[0]), xs, ys),
    )
    layers = layer_fn(model, opt)
    out['layers'] = [dict(cov=L['cov'].tolist(), spec=spectrum(L['h']), neg=(L['h'] < 0).float().mean().item(),
                          below=(L['h'] < cfg['weight_decay']).float().mean().item(), shape=list(p.shape))
                     for L, p in zip(layers, weight_matrices(model))]
    out['slices'] = slices(model, layers, X, y, kind)
    out['paths'] = sample_paths(model, opt, X, y)
    return out


@torch.no_grad()
def sample_paths(model, opt, X, y, k=4, n=17):
    """Train loss along k random posterior-sample directions: mean + t * (sample - mean),
    t in [-2, 2], so t = +-1 is a posterior sample."""
    params = list(model.parameters())
    base = [p.detach().clone() for p in params]
    out = []
    for _ in range(k):
        with opt.sampled_params(train=False):
            d = [p.detach() - b for p, b in zip(params, base)]
        losses = []
        for t in torch.linspace(-2, 2, n):
            for p, b, di in zip(params, base, d):
                p.copy_(b + t * di)
            losses.append(F.binary_cross_entropy_with_logits(model(X), y).item())
        for p, b in zip(params, base):
            p.copy_(b)
        out.append(losses)
    return out


@torch.no_grad()
def slices(model, layers, X, y, kind, n=25):
    """Train loss along the largest- and smallest-variance posterior directions,
    over +-3 posterior standard deviations."""
    Ws = weight_matrices(model)
    best = {'max': None, 'min': None}
    for li, (L, p) in enumerate(zip(layers, Ws)):
        s2 = L['s2'] if kind == 'evon' else L['s2'].view(p.shape)
        flat = s2.flatten()
        for which, idx in [('max', flat.argmax()), ('min', flat.argmin())]:
            v = flat[idx].item()
            if best[which] is None or (v > best[which][0] if which == 'max' else v < best[which][0]):
                best[which] = (v, li, divmod(idx.item(), s2.shape[1]))
    out = {}
    ts = torch.linspace(-3, 3, n)
    for which, (v, li, (k, l)) in best.items():
        p = Ws[li]
        if kind == 'evon':
            d = direction(layers[li], k, l, p.shape)
        else:
            d = torch.zeros_like(p); d[k, l] = 1.0
        sd = math.sqrt(v)
        base = p.detach().clone()
        losses = []
        for t in ts:
            p.copy_(base + t * sd * d)
            losses.append(F.binary_cross_entropy_with_logits(model(X), y).item())
        p.copy_(base)
        out[which] = dict(layer=li, sd=sd, loss=losses)
    return out


def run(args):
    idx, cfg = args
    torch.set_num_threads(1)
    X, y = train_data()
    Xt, yt = test_data(400)
    Xf = far_data(300)
    _, _, Gc = grid(*COARSE)
    torch.manual_seed(0)  # same init and noise stream for every config
    model = MLP()
    opt = build_opt(model, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=0.0, T_max=STEPS)
    ess, wd = cfg['ess'], cfg['weight_decay']
    ck = set(CKPTS)
    curves = dict(step=[], train_loss=[], test_nll=[], mi_far=[], layer_sd=[], below=[])
    snaps = []  # coarse predictive at each checkpoint
    t0 = time.time()
    diverged_at = None
    run_loss, run_n = 0.0, 0

    def checkpoint(step):
        lt, lf, lg = sample_logits(model, opt, [Xt, Xf, Gc], 8)
        pt = torch.sigmoid(lt).mean(0).clamp(1e-6, 1 - 1e-6)
        ytf = yt.squeeze(-1)
        P = torch.sigmoid(lf)
        sds, bel = [], []
        for p in weight_matrices(model):
            h = opt.state[p]['h_mom'] if 'h_mom' in opt.state[p] else torch.full_like(p, cfg['hess_init'])
            sds.append((1 / (ess * (h + wd))).mean().sqrt().item())
            bel.append((h < wd).float().mean().item())
        curves['step'].append(step)
        curves['train_loss'].append(run_loss / max(run_n, 1))
        curves['test_nll'].append(-(ytf * pt.log() + (1 - ytf) * (1 - pt).log()).mean().item())
        curves['mi_far'].append((entropy(P.mean(0)) - entropy(P).mean(0)).mean().item())
        curves['layer_sd'].append(sds)
        curves['below'].append(bel)
        snaps.append(torch.sigmoid(lg).mean(0).tolist())

    checkpoint(0)
    for step in range(1, STEPS + 1):
        with opt.sampled_params(train=True):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(X), y)
            loss.backward()
        opt.step()
        sched.step()
        run_loss += loss.item(); run_n += 1
        if not math.isfinite(run_loss) or (step % 50 == 0 and (not finite_state(opt, wd) or
                                                              not all(torch.isfinite(p).all() for p in model.parameters()))):
            diverged_at = step
            break
        if step in ck:
            checkpoint(step)
            run_loss, run_n = 0.0, 0
    out = dict(cfg=cfg, idx=idx, diverged_at=diverged_at, curves=curves, snaps=snaps, train_seconds=time.time() - t0)
    if diverged_at is None:
        t1 = time.time()
        out['final'] = evaluate(model, opt, cfg, lambda m, o: [evon_layer(o, p, ess, wd) for p in weight_matrices(m)], 'evon')
        out['eval_seconds'] = time.time() - t1
    path = os.path.join(OUT, f'{idx}.json.gz')
    with gzip.open(path + '.tmp', 'wt') as f:
        json.dump(out, f)
    os.replace(path + '.tmp', path)
    return idx, out['train_seconds'], out.get('eval_seconds'), diverged_at


# ---------------------------------------------------------------------------
# References: notebook 2's two IVON cells (EVON's notebook cell is in the grid)
# ---------------------------------------------------------------------------

IVON_REFS = {
    'ivon_iso': dict(lr=0.7, ess=5e1, weight_decay=1e-5, hess_init=1.0, beta1=0.9, beta2=1.0, clip_radius=0.5),
    'ivon_diag': dict(lr=0.02, ess=2e4, weight_decay=1e-5, hess_init=4e-3, beta1=0.9, beta2=0.9995, clip_radius=0.1),
}


def ivon_layers(model, opt, n=40):
    g = opt.param_groups[0]
    var = 1.0 / (g['ess'] * (g['hess'].double() + g['weight_decay']))
    hess = g['hess'].double()
    out, off = [], 0
    for p in model.parameters():
        if p.ndim == 2:
            aa, bb = strided_entries(p, n)
            v = var[off:off + p.numel()]
            out.append(dict(cov=torch.diag(v.view(p.shape)[aa, bb]), h=hess[off:off + p.numel()], s2=v))
        off += p.numel()
    return out


def run_ivon(name):
    from ivon import IVON
    torch.set_num_threads(1)
    hp = IVON_REFS[name]
    X, y = train_data()
    torch.manual_seed(0)
    model = MLP()
    opt = IVON(model.parameters(), rescale_lr=False, **hp)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=0.0, T_max=STEPS)
    for _ in range(STEPS + 1):
        with opt.sampled_params(train=True):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(X), y).backward()
        opt.step()
        sched.step()
    cfg = dict(weight_decay=hp['weight_decay'])
    final = evaluate(model, opt, cfg, ivon_layers, 'ivon')
    return name, dict(hp=hp, final=final)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfgs = configs()
    assert len(cfgs) == 5184, len(cfgs)
    done = {int(f.split('.')[0]) for f in os.listdir(OUT) if f.endswith('.json.gz')}
    todo = [(i, c) for i, c in enumerate(cfgs) if i not in done]
    # interleave so slow (frequency 1, whitened) and fast runs mix across workers
    todo.sort(key=lambda ic: (ic[0] * 7919) % len(cfgs))
    workers = int(os.environ.get('WORKERS', 64))
    print(f'{len(cfgs)} configs, {len(done)} done, {len(todo)} to run on {workers} workers', file=sys.stderr, flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for k, (idx, ts, es, dv) in enumerate(pool.imap_unordered(run, todo)):
            if k % 100 == 0:
                print(f'[{k}/{len(todo)}] {time.time() - t0:.0f}s  last: train {ts:.0f}s eval {es or 0:.0f}s'
                      f'{" diverged@" + str(dv) if dv else ""}', file=sys.stderr, flush=True)
    refs_path = os.path.join(HERE, 'nn_refs.json.gz')
    if not os.path.exists(refs_path):
        with Pool(2) as pool:
            refs = dict(pool.map(run_ivon, list(IVON_REFS)))
        with gzip.open(refs_path, 'wt') as f:
            json.dump(refs, f)
    print(f'done in {time.time() - t0:.0f}s', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
