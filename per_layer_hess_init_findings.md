# Per-layer automatic `hess_init` in EVON: four-rule comparison

Investigation for removing `hess_init` as a user-facing knob. The goal is a
per-layer rule that keeps initialization stable until EVON learns the actual
curvature, leaving ESS as the only global control on posterior sampling noise.

Notebook: `2b_hess_init.ipynb`, section 5 (§1–4 are the earlier diagnosis).
Raw per-layer results: `per_layer_hess_init_results.csv` (60 runs = 4 rules ×
widths {32,64,128,256,512} × seeds {0,1,2}; per-run diagnostics in the dev
driver, not shipped).

## Verified sampling formula

From `evon.py` `_sample_params`:

```text
denom = sqrt(ess * (h_mom + weight_decay))
noise = normal() / denom
```

`Q` is orthogonal, so for uniform `h` every weight coordinate gets exactly

```text
sigma0 = 1 / sqrt(ess * (h0 + weight_decay))
```

`weight_decay` adds to `h` in both the noise scale and the Newton denominator;
`eps` only damps the mean preconditioner. Effective posterior variance is
`1 / (ess * (h + wd))`.

## Rules

The automatic rule applies to 2-D weight matrices only; biases keep the scalar
`hess_init = 4e-3`. Each rule has one constant calibrated once at `w=128`,
seed 2, so the middle W2 layer reproduces the shipped
`sigma0 = 0.09117` (`hess_init=4e-3`, `ess=3e4`). No tuning per width, layer or
seed.

| rule | `h0 + wd` per weight matrix | needs |
| --- | --- | --- |
| A fixed | `4e-3` | nothing |
| B init-variance / fan-in | `c_B / Var_init(W)`, `Var_init = 1/(3 fan_in)` | architecture |
| C weight-RMS | `c_C / mean(W^2)` | initialized weights |
| D activation | `c_D * E[||x_l||^2] / E[||y_l||^2]` | eval-mode activations on a fixed batch |

Calibrated constants: `c_B = 1.044e-5`, `c_C = 1.041e-5`, `c_D = 1.420e-3`.

Two facts: under torch's default `Linear` init,
`mean(W^2) ≈ Var_init = 1/(3 fan_in)`, so **B and C are the same rule up to
init-sampling noise**. For a linear layer
`E[||x||^2]/E[||y||^2] = in / (out * mean(W^2))`, so on square middle layers
**D coincides with C**, and differs only at W1 (`in/out = 2/w`) and W5
(`in/out = w`), where it is a single-output high-variance estimator. The
brief's activation formula drops the `out` factor in
`E[||dy||^2] = out * sigma0^2 * E[||x||^2]`; the measured `||dy||/||y||`
exposes it.

All other settings held fixed: `ess=3e4`, `lr=1e-3`, `betas=(0.9, 0.9995)`,
`weight_decay=1e-5`, cosine schedule to 0, 7500 steps, same pinned data.

## Results

Initial relative noise `rho0 = sigma0 / RMS(W)`, over all layers and widths:

| rule | mean | CV | range |
| --- | --- | --- | --- |
| A | 1.654 | 0.69 | 0.223 – 3.581 |
| B | 1.794 | 0.01 | 1.731 – 1.859 |
| C | 1.789 | 0.00 | 1.789 – 1.789 |
| D | 1.950 | 0.65 | 0.091 – 4.463 |

Fraction of seeds reaching loss `< 1e-4`, and median steps to reach it:

| width | A | B | C | D |
| --- | --- | --- | --- | --- |
| 32 | 0/3 | 2/3 | 2/3 | 1/3 |
| 64 | 1/3 | 2/3 | 3/3 | 2/3 |
| 128 | 2/3 | 3/3 | 3/3 | 2/3 |
| 256 | 3/3 | 3/3 | 2/3 | 3/3 |
| 512 | 3/3 | 2/3 | 3/3 | 1/3 |

Median steps to `< 1e-4`:

| width | A | B | C | D |
| --- | --- | --- | --- | --- |
| 32 | 4137 | 702 | 469 | 2745 |
| 64 | 2364 | 713 | 598 | 1666 |
| 128 | 1487 | 810 | 967 | 2177 |
| 256 | 462 | 1410 | 1238 | 1722 |
| 512 | 30 | 2149 | 2008 | 2562 |

- **Invariance.** A and D sweep `rho0` over more than a decade. B is flat at
  `1.79 ± 0.05`; C is exactly `1.789` in every layer and width.
- **D is worse, not better.** It assigns W5 (one output) `h0 = 0.005 – 17.8`
  across seeds, so W5's `rho0 = 0.09 – 0.43`. At `w=512` W5 ends with
  `mean_h ≈ 13 – 35` and posterior variance `3e-5` — a collapsed output
  posterior, loss `0.003 – 0.019` on 2/3 seeds. Measured
  `||dy||/||y||` is `0.13 – 0.16` at W5 versus `≈3.5` in the middle layers,
  so it does not equalize relative output noise once the norm is summed over
  outputs.
- **Fit vs speed.** No rule is unstable in the way a sub-floor scalar
  `hess_init` was (§2). A over-damps narrow nets (0/3 reach `1e-4` at `w=32`,
  median loss `1.8e-3`) but is fastest at `w=512` (30 steps). B/C remove the
  narrow-width failure but need `~2000` steps at `w=512`: constant `rho` forces
  `h0 ∝ fan_in`, so the Newton step `lr/(h0+wd)` shrinks with width. D is
  slowest almost everywhere.
- **The initial level persists.** At `w=128` the middle layers end at
  `r = ||h - h0|| / (sqrt(P) h0)` of only 1–3, with `mean_h` just
  `1.6 – 4.5×` the decayed init envelope `h0 * beta2^7500`, i.e. the initial
  level is still ~20–60% of the learned value. W1/W5 are fully learned
  (`r ≈ 4 – 600`). Deriving `h0` per layer matters at the fixed budget.

## Failure cases

- **B**: `w=32` seed 1 ratchets W1 to `mean_h = 0.36` (~7e3× its `h0`) and
  stalls at `5.4e-2`; `w=512` seed 2 is still at `0.26` after 7500 steps.
- **C**: `w=256` seed 2 reaches `1e-6` by step ~3000, then the Price estimate
  drifts and it ends at `1.1e-2`; `w=512` seed 2 is slow (`3.4e-6`).
- **D**: adds output-layer collapse on top of the above; underfits at
  `w=32` (3e-2), `w=64` (1e-2), `w=128` (2.3e-3), and `w=512` (1.9e-2, 3.3e-3).
- These are near-separation, draw-dependent Price-ratchet hazards already seen
  in §3–4. No rule removes them; D has the most.

## Recommendation

Adopt the **weight-RMS rule C**:

```text
h0_l = max( c / mean(W_l^2) - weight_decay, 0 )     (2-D weights only)
c = 1.04e-5                                         (ess = 3e4, rho = 1.789)
```

It is exactly `rho`-invariant by construction, absorbs any init scheme (He,
Xavier, custom), and needs no forward pass. It did not lose to the fan-in or
activation alternatives.

The **fan-in rule B** (`c_B` above) is an equally good closed form for the
default init if reading the weights at initialization is undesirable.

Deploying the **activation rule D is not worth the extra complexity**: it is
numerically C on square layers and strictly worse at the edges, and its output
layer estimate is high-variance.

## Can `hess_init` be removed from the public API?

Yes, as a *required* argument. Default it to the per-tensor derivation above
and keep a scalar override for back-compat.

Bookkeeping that keeps ESS as the global noise knob: the constant `c` must be
**fixed**. Then

```text
sigma0 = sqrt(mean(W^2) / (ess * c))
```

so changing `ess` scales the injected noise globally. If instead `c` is
re-derived from `ess` (the const-`rho` framing), `ess` cancels and no longer
controls the noise magnitude.

Caveats: `c` (or equivalently `rho`) is worth re-calibrating per `ess`/task
family, and the near-separation failures above persist regardless of the rule.
No single rule eliminates the seed fragility of this setup.

## Reproduce

The four-rule sweep is generated by the dev driver
(`driver_rules.py`, checkpointed shards) and shipped as
`per_layer_hess_init_results.csv`. Notebook section 5 recomputes the
calibration constants and renders the tables/figures directly from that CSV.

```text
2b_hess_init.ipynb       section 5 (four-rule comparison) + findings addendum
per_layer_hess_init_results.csv   300 per-layer rows, 60 runs
```
