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
  `h0 ∝ fan_in`, so the middle layers' Newton step `lr/(h0+wd)` shrinks with
  width **when whitening is off — the shipped setting**. With
  `whiten_prec_grad=True` the two rules become equivalent; see the mechanism
  check below. D is slowest almost everywhere.
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

## Mechanism check: whitening vs effective learning rate (PR #2 review)

Reviewer note: with `whiten_prec_grad=True` (EVON's default), Newton–Schulz
whitening is approximately scale-invariant, so a uniform scalar `h0` should
mostly rescale the pre-whitened update rather than shrink the final step; the
wide-width slowdown of the RMS rule would then not be an `lr/(h0+wd)` effect.
The shipped runs, however, set `whiten_prec_grad=False` explicitly (not the
default), so the concern does not apply to them as written. To settle it we
bracketed both settings.

Matrix: `w in {128,512}` × `{A fixed, C RMS}` × `sampling {on, off}` ×
`whiten_prec_grad {False,True}` × seeds {0,2}. Deterministic mode =
`disable_sampling()` with no `sampled_params` context (h adapts by the `g^2`
EMA fallback, no noise). Raw data: `whiten_diag.csv`.

Static first-step probe (seed-2 init, full-batch gradient, per-layer
`||update||`, before any adaptation):

| width | layer | raw A/C | whitened A/C |
| --- | --- | --- | --- |
| 512 | W2–W4 | 3.97 – 3.99 | 1.000 |
| 512 | W1 | 0.02 | 1.000 |
| 512 | W5 | 3.95 | 1.000 |
| 128 | all | 0.02 – 1.01 | 1.000 |

With whitening on the per-layer update magnitude is *identical* for A and C
(ratio 1.000), exactly as argued. With whitening off (shipped), the middle
layers' raw update is 4× larger for A at `w=512`, while C's W1 gets a 64×
larger step (`h0=5.3e-5`) — a redistribution, not a uniform slowdown. Middle
layers dominate the function, so C trains slower at wide width.

Training, steps to `1e-4` (two seeds):

| width | whiten | sampling | A | C |
| --- | --- | --- | --- | --- |
| 128 | off | off | 1791 / 1871 | 1685 / 1669 |
| 128 | off | on | 1487 / 2222 | 715 / 1102 |
| 128 | on | off | 2661 / 2727 | 2690 / 2741 |
| 128 | on | on | fail 0.29–0.44 | fail 0.28–0.44 |
| 512 | off | off | 1432 / 1526 | 4721 / 4744 |
| 512 | off | on | 30 / 104 | 1840 / 2167 |
| 512 | on | off | 2110 / 2147 | 2089 / 2103 |
| 512 | on | on | fail 0.73 / 0.94 | fail 0.36 / 0.90 |

- **Whitening on:** A and C are indistinguishable — deterministic final losses
  `1.2e-5` vs `1.2e-5` at `w=128` and `7.7e-6` vs `7.6e-6` at `w=512`, and the
  sampling runs fail identically. Confirms the scaling argument.
- **Whitening off (shipped):** the gap is real. At `w=512` A reaches `1e-4` in
  ~1450 steps deterministically (C ~4730; ratio 3.3, matching the 4× raw
  update) and in 30–104 steps with sampling (C ~2000). So in the shipped
  regime the speed gap is an effective-step effect, with posterior noise as a
  secondary channel: fixed `h0` gives `rho=3.58` at `w=512` vs C's 1.79, and
  A's mean displacement is ~10× C's.
- **Whitening is not usable on this task anyway:** with sampling it fails to
  fit (loss 0.3–0.9) at both widths, and deterministically it is ~1.5× slower
  than no whitening. That is why the config disables it.

Net: the original wording was correct *for the config actually used*
(`whiten_prec_grad=False`) but over-general. Qualified finding: the RMS rule
slows wide models through the effective step only when whitening is off; with
whitening on, A and C are equivalent.

## `beta2` wash-out sweep (PR #2 follow-up)

Follow-up suggestion: keep the RMS-derived `h0` but lower `beta2` so the
initialization is forgotten faster, rather than adding a burn-in or another
`h0` rule. With `beta2=0.9995` the envelope `h0 * beta2^t` barely decays over
7500 steps; at wide layers the RMS `h0` then holds the mean update down.

Sweep (sampled EVON, `whiten_prec_grad=False`): `w in {128,512}` × `{A,C}` ×
`beta2 in {0.9995,0.999,0.99,0.9}` × 2 seeds. Data: `beta_sweep.csv`.

Steps to loss `< 1e-4` (median of 2 seeds):

| width | beta2 | A fixed | C RMS |
| --- | --- | --- | --- |
| 128 | 0.9995 | 1854 | 908 |
| 128 | 0.999 | 1424 | 760 |
| 128 | 0.99 | 792 | 628 |
| 128 | 0.9 | 2696 | 2746 |
| 512 | 0.9995 | 67 | 2004 |
| 512 | 0.999 | 121 | 1073 |
| 512 | 0.99 | 296 | 462 |
| 512 | 0.9 | 1800 | 1857 |

Middle-layer departure from the envelope `beta2^t * h0` (first step where
`mean_h > 2 * h0 * beta2^t`; `never` otherwise), median over W2–W4:

| width | beta2 | A | C |
| --- | --- | --- | --- |
| 512 | 0.9995 | 100 | never |
| 512 | 0.999 | 3300 | 4050 |
| 512 | 0.99 | 100 | 400 |
| 512 | 0.9 | 100 | 100 |
| 128 | 0.9995 | 5450 | 6950 |
| 128 | 0.99 | 200 | 200 |

Findings:

- **`beta2` is the lever.** At `w=512` the RMS rule's steps to `1e-4` fall
  from 2003 (`0.9995`) to 1073 (`0.999`) to **462** (`0.99`), recovering
  fixed's speed while keeping the width-normalized init. At `w=128` RMS is
  faster than fixed at every `beta2`.
- **The mechanism is the envelope lifetime.** At `w=512`, `beta2=0.9995` the
  RMS middle layers never leave `2 * h0 * beta2^t` in 7500 steps; at
  `beta2=0.99` they depart by step ~400.
- **Fit is preserved** at `0.99`/`0.999`: RMS final losses 0.0 / 0.0 at
  `w=512`, `1e-38`–0 at `w=128`.
- **The cost is Price-estimate noise.** The late relative jitter of `mean_h`
  rises from ~0.14 (`0.9995`) to ~1.6 (`0.99`) to ~2.1 (`0.9`), and W1's `h`
  estimate inflates by ~3 orders of magnitude over that range (RMS `w=512`
  W1: `2.9e-3 → 0.16 → 3.97 → 37`). At `beta2=0.9` it is too noisy: one RMS
  `w=512` seed fails (4.6e-2) and fixed underfits `w=128` (1.5e-3–6e-3).

Conclusion: `beta2≈0.99` is a simpler and effective fix for the persistence
problem than a new `h0` rule or a burn-in, and it preserves the intended EVON
coupling — but it trades away Price-estimate stability. The original
`beta2=0.9995` was a large part of why the RMS rule looked slow at wide width.

### Does `beta2=0.99` help fixed too?

The 128/512 sweep suggests yes; to test fixed's original weakness
(narrow-width over-damping) we ran all widths at `beta2=0.99`, 3 seeds
(`beta99_sweep.csv`). Median final loss (worst seed):

| width | A fixed @0.99 | A fixed @0.9995 | C RMS @0.99 | C RMS @0.9995 |
| --- | --- | --- | --- | --- |
| 32 | 7.4e-6 (2.2e-5) | 1.8e-3 (2.8e-3) | 2.5e-11 (3.6e-9) | 9.8e-5 (1.9e-4) |
| 64 | 3.2e-19 (2.5e-14) | 3.4e-4 (5.3e-2) | 8.9e-19 (2.5e-13) | 8.9e-15 (1.4e-6) |
| 128 | 2.8e-38 (8.0e-31) | 1.4e-8 (3.8e-2) | 8.9e-38 (3.2e-21) | 3.5e-27 (7.4e-16) |
| 256 | 0.0 (0.0) | 9.0e-31 (8.1e-8) | 1.9e-38 (5.6e-38) | 1.9e-8 (1.1e-2) |
| 512 | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (3.4e-6) |

At `beta2=0.99` **all 3 seeds reach `<1e-4` at every width for both rules**.
Lowering `beta2` removes fixed's narrow-width over-damping (w32 median
`1.8e-3 → 7.4e-6`). Steps to `1e-4` at `0.99`, A vs C: 2696 vs 1278 (w32),
1545 vs 750 (w64), 809 vs 702 (w128), 360 vs 534 (w256), 282 vs 457 (w512) —
RMS faster narrow, fixed faster wide. The costs apply to both: W1 `h`
inflation (A 0.9–20, C 1.4–5.9) and middle-layer jitter ~0.8–1.8.

So `beta2=0.99` is not RMS-specific: it is an alternative fix for the same
persistence problem, and it makes fixed robust across widths too. At
`beta2=0.99` the choice between fixed and RMS is a narrow-vs-wide speed
trade-off rather than a stability one.

## Reproduce

The four-rule sweep is generated by the dev driver
(`driver_rules.py`, checkpointed shards) and shipped as
`per_layer_hess_init_results.csv`. Notebook section 5 recomputes the
calibration constants and renders the tables/figures directly from that CSV.

```text
2b_hess_init.ipynb       section 5 (four-rule comparison), 5b (whitening),
                         5c (beta2 wash-out) + findings addendum
per_layer_hess_init_results.csv   300 per-layer rows, 60 runs
whiten_diag.csv                    32 runs, whitening/effective-step check
beta_sweep.csv                     32 runs, beta2 wash-out sweep (w 128/512)
beta99_sweep.csv                   30 runs, beta2=0.99 all widths, A and C
```
