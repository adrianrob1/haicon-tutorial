"""Pack nn_runs/ into evon_sweep/site/nn.html + site/nn_data/shard<k>.js.

nn.html carries the per-run summary metrics of the whole sweep and the IVON
references; each shard holds the heavy per-run payload (predictive grids,
sampled boundaries, per-layer covariance, spectra, curves) for a contiguous
block of runs, base64-encoded binary so it stays small and loads from disk too.
"""

import base64
import gzip
import json
import math
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nn_sweep as S  # noqa: E402

SITE = os.path.join(HERE, 'site')
RUNS = os.path.join(HERE, 'nn_runs')
N_SNAPS = 12  # coarse predictive snapshots kept for the training scrubber
COV_DECADES = 6  # covariance stored as sign * log10(|v| / vmax) over this many decades, int8


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def u8(x, lo=0.0, hi=1.0):
    return np.clip(np.round((np.asarray(x, dtype=np.float64) - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


def cov_i8(C):
    C = np.asarray(C, dtype=np.float64)
    vmax = float(np.abs(C).max()) or 1.0
    mag = np.log10(np.maximum(np.abs(C) / vmax, 10.0 ** -COV_DECADES)) / COV_DECADES + 1  # 0..1
    q = np.where(np.abs(C) > 0, np.sign(C) * np.round(mag * 127), 0).astype(np.int8)
    return q, vmax


def boundaries(lines_per_sample):
    lens, flat = [], []
    for lines in lines_per_sample:
        lens.append([len(ln) for ln in lines])
        for ln in lines:
            flat.extend(v for pt in ln for v in pt)
    return lens, b64(np.round(np.asarray(flat, dtype=np.float64) * 1000).astype(np.int16))


def pack_final(f):
    covs, vmaxs = zip(*(cov_i8(L['cov']) for L in f['layers']))
    blens, bflat = boundaries(f['boundary_samples'])
    mlens, mflat = boundaries([f['boundary_mean']])
    return dict(
        p=b64(u8(f['pbar'])),
        m=b64(u8(f['mi_grid'], 0, math.log(2))),
        bl=blens, bc=bflat, ml=mlens[0], mc=mflat,
        cv=b64(np.concatenate([c.reshape(-1) for c in covs])), cn=[int(np.asarray(L['cov']).shape[0]) for L in f['layers']],
        vm=[float(f'{v:.4g}') for v in vmaxs],
        h=b64(np.asarray([L['spec'] for L in f['layers']], dtype=np.float32)),
        sn=b64(np.asarray(f['sample_nll'], dtype=np.float32)),
        sl={k: dict(layer=v['layer'], sd=float(f'{v["sd"]:.4g}'), loss=b64(np.asarray(v['loss'], dtype=np.float32))) for k, v in f['slices'].items()},
        shapes=[L['shape'] for L in f['layers']],
        **({'pa': b64(np.asarray(f['paths'], dtype=np.float32))} if 'paths' in f else {}),
    )


def summary(r):
    f = r.get('final')
    if f is None:
        return dict(d=r['diverged_at'])
    hidden = [L['below'] for L in f['layers'][1:4]]
    sn = f['sample_nll']
    return dict(
        d=0,
        auroc=f['auroc'], test_nll=f['test_nll'], test_acc=f['test_acc'], ece=f['ece'],
        mi_far=f['mi_far'], mi_test=f['mi_test'],
        sample_gap=sum(sn) / len(sn) - f['train_nll_mean'],
        below=sum(hidden) / len(hidden),
        neg=max(L['neg'] for L in f['layers']),
    )


def pack_run(r, snap_idx):
    c = r['curves']
    n = len(c['step'])
    cur = np.full((n, 3 + 5 + 5), np.nan, dtype=np.float32)
    cur[:, 0] = c['train_loss']; cur[:, 1] = c['test_nll']; cur[:, 2] = c['mi_far']
    cur[:, 3:8] = np.asarray(c['layer_sd']); cur[:, 8:13] = np.asarray(c['below'])
    snaps = np.full((N_SNAPS, len(r['snaps'][0])), 0, dtype=np.uint8)
    have = []
    for j, si in enumerate(snap_idx):
        if si < len(r['snaps']):
            snaps[j] = u8(r['snaps'][si]); have.append(j)
    out = dict(n=n, c=b64(cur), s=b64(snaps), sh=len(have))
    if r.get('final'):
        out['f'] = pack_final(r['final'])
    return out


def load(i):
    with gzip.open(os.path.join(RUNS, f'{i}.json.gz'), 'rt') as f:
        return json.load(f)


def r5(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(f'{x:.5g}')


def work(args):
    """Load + pack one shard in a worker (the JSON parsing dominates)."""
    k, idxs, snap_idx = args
    runs, summ = [], {}
    for i in idxs:
        r = load(i)
        runs.append(pack_run(r, snap_idx))
        summ[i] = summary(r)
    return k, runs, summ


def main():
    cfgs = S.configs()
    have = {int(f.split('.')[0]) for f in os.listdir(RUNS) if f.endswith('.json.gz')}
    missing = [i for i in range(len(cfgs)) if i not in have]
    if missing and '--partial' not in sys.argv:
        sys.exit(f'{len(missing)} runs missing; run nn_sweep.py to finish (or pass --partial)')
    keys = list(S.GRID)
    per_shard = len(S.GRID['hess_init']) * len(S.GRID['ess']) * len(S.GRID['beta2']) * len(S.GRID['beta1']) \
        * len(S.GRID['precondition_frequency']) * len(S.GRID['beta3'])  # contiguous block: whiten x lr x wd fixed
    n_shards = len(cfgs) // per_shard
    steps = S.CKPTS
    snap_idx = sorted(set(np.round(np.linspace(0, len(steps) - 1, N_SNAPS)).astype(int).tolist()))
    assert len(snap_idx) == N_SNAPS

    os.makedirs(os.path.join(SITE, 'nn_data'), exist_ok=True)
    jobs = [(k, [i for i in range(k * per_shard, (k + 1) * per_shard) if i in have], snap_idx) for k in range(n_shards)]
    summ = {}
    sizes = []
    with Pool(min(24, os.cpu_count())) as pool:
        for k, runs, s in pool.imap_unordered(work, jobs):
            summ.update(s)
            idxs = jobs[k][1]
            body = json.dumps(dict(idx=idxs, runs=runs), separators=(',', ':'))
            path = os.path.join(SITE, 'nn_data', f'shard{k}.js')
            with open(path, 'w') as f:
                f.write(f'__nnshard({k},{body});\n')
            sizes.append(os.path.getsize(path) / 1e6)

    cols = ['auroc', 'test_nll', 'test_acc', 'ece', 'mi_far', 'mi_test', 'sample_gap', 'below', 'neg']
    summary_cols = {c: [r5(summ[i].get(c)) if i in summ else None for i in range(len(cfgs))] for c in cols}
    summary_cols['d'] = [summ[i]['d'] if i in summ else -1 for i in range(len(cfgs))]  # -1 = not run yet

    with gzip.open(os.path.join(HERE, 'nn_refs.json.gz'), 'rt') as f:
        refs = json.load(f)
    nb_idx = cfgs.index(S.NB)
    xs, ys, _ = S.grid(*S.FINE)
    data = dict(
        grid=S.GRID, keys=keys, base_lr={'true': S.BASE_LR[True], 'false': S.BASE_LR[False]},
        steps=steps, snap_steps=[steps[i] for i in snap_idx], per_shard=per_shard, nb=nb_idx,
        fine=list(S.FINE), coarse=list(S.COARSE), gx=list(S.GX), gy=list(S.GY), cov_decades=COV_DECADES,
        train=dict(X=S.train_data()[0].tolist(), y=S.train_data()[1].view(-1).tolist()),
        summary=summary_cols,
        refs={k: dict(hp=v['hp'], s=summary(dict(final=v['final'], diverged_at=None)), f=pack_final(v['final'])) for k, v in refs.items()},
    )
    with open(os.path.join(HERE, 'template.html')) as f:
        style = re.search(r'<style>(.*?)</style>', f.read(), re.S).group(1)
    with open(os.path.join(HERE, 'nn_template.html')) as f:
        html = f.read()
    html = html.replace('/*__STYLE__*/', style).replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    with open(os.path.join(SITE, 'nn.html'), 'w') as f:
        f.write(html)
    open(os.path.join(SITE, '.nojekyll'), 'w').close()
    print(f'wrote {SITE}/nn.html ({os.path.getsize(os.path.join(SITE, "nn.html")) / 1e6:.1f} MB), {n_shards} shards '
          f'{min(sizes):.1f}-{max(sizes):.1f} MB, {len(have)}/{len(cfgs)} runs, {len(missing)} missing')


if __name__ == '__main__':
    main()
