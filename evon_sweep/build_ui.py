"""Pack evon_sweep/sweep.json into evon_sweep/site/: index.html carries the
per-run summary metrics for the whole sweep; the per-step trajectories live in
one data/slice<k>.js per (phasing, beta3) slice and are loaded on demand.

Shards are JS (``__shard(k, {...})``) rather than JSON so the page also works
when opened straight from disk, where fetch() is blocked."""

import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, 'site')
SLICE_KEYS = ['phasing', 'beta3']  # the last grid axes, so slice = idx % n_slices


def r4(x):
    return None if x is None else float(f'{x:.5g}')


def dumps(x):
    return json.dumps(x, separators=(',', ':'))


def main():
    with open(os.path.join(HERE, 'sweep.json')) as f:
        S = json.load(f)

    keys = list(S['grid'])
    assert keys[-len(SLICE_KEYS):] == SLICE_KEYS, keys
    n_slices = 1
    for k in SLICE_KEYS:
        n_slices *= len(S['grid'][k])

    runs = sorted(S['runs'], key=lambda r: r['idx'])
    assert [r['idx'] for r in runs] == list(range(len(runs)))
    steps = [t[0] for t in max(runs, key=lambda r: len(r['traj']))['traj']]

    summary = {
        'd': [int(r['diverged']) for r in runs],
        **{short: [r4(r[long]) for r in runs] for short, long in
           [('kl', 'kl_exact'), ('gap', 'kl_gap'), ('klg', 'kl_gauss'), ('loss', 'loss'), ('corr', 'corr')]},
    }
    data = dict(
        setup=S['setup'],
        grid=S['grid'],
        keys=keys,
        nslices=n_slices,
        exact=S['exact'],
        laplace=S['laplace'],
        ivon={m: dict(t=[r4(v) for t in iv['traj'] for v in t[1:]], klt=[r4(v) for v in iv['klt']]) for m, iv in S['ivon'].items()},
        steps=steps,
        summary=summary,
    )

    # only this page's outputs; nn_build.py owns nn.html and nn_data/
    shutil.rmtree(os.path.join(SITE, 'data'), ignore_errors=True)
    os.makedirs(os.path.join(SITE, 'data'))
    with open(os.path.join(HERE, 'template.html')) as f:
        html = f.read()
    with open(os.path.join(SITE, 'index.html'), 'w') as f:
        f.write(html.replace('/*__DATA__*/null', dumps(data)))
    open(os.path.join(SITE, '.nojekyll'), 'w').close()

    sizes = []
    for k in range(n_slices):
        part = runs[k::n_slices]  # base-config order within the slice
        shard = dict(
            t=[[r4(v) for t in r['traj'] for v in t[1:]] for r in part],
            klt=[[r4(v) for v in r['klt']] for r in part],
        )
        path = os.path.join(SITE, 'data', f'slice{k}.js')
        with open(path, 'w') as f:
            f.write(f'__shard({k},{dumps(shard)});\n')
        sizes.append(os.path.getsize(path) / 1e6)
    print(f'wrote {SITE}: index.html {os.path.getsize(os.path.join(SITE, "index.html")) / 1e6:.1f} MB, '
          f'{n_slices} shards {min(sizes):.1f}-{max(sizes):.1f} MB')


if __name__ == '__main__':
    main()
