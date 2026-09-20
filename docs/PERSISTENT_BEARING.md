# PersistentBearing Development Track

## Baseline Protection

The reproduced baseline is preserved in two ways:

- Git branch: `baseline-reproduction-20260720`
- Baseline commit: `5673943`

The existing model and entrypoints are unchanged:

```bash
bash scripts/cvphr_train_paper_baseline.sh 0
BESTPTH_DIR=/path/to/baseline/result bash scripts/cvphr_test_paper_baseline.sh
```

PersistentBearing is developed on branch `persistent-bearing`. It writes only to:

```text
results/persistent_bearing/
log/persistent_bearing/
```

To inspect the original source after committing or stashing current work:

```bash
git switch baseline-reproduction-20260720
```

## Implemented Scope

The current first stage contains:

1. Python 3.9-compatible DINOv3 ConvNeXt-Small and ConvNeXt-Base loaders.
2. Shared frozen dense backbone with UAV/satellite domain adapters.
3. Topologically correct stitching of the four satellite patches.
4. A dense joint `C(x, y, theta)` correlation volume.
5. Joint soft-label SE(2) NLL and calibrated top-K pose hypotheses.
6. Hypothesis-conditioned tentative/confirmed online memory.
7. Independent train and test programs with separate checkpoints.
8. An optional large model with domain-specific and shared residual MLPs.
9. An optional symmetry-aware circular (SAC) heading head with a multimodal
   von Mises distribution, continuous within-bin refinement, concentration,
   antipodal symmetry score, and calibrated heading confidence.

## Symmetry-Aware Circular Heading

The SAC variant leaves the position path unchanged. It marginalizes the
existing cost volume over position, refines the circular angle evidence with
periodic 1D convolutions, and represents every angle bin as a von Mises
component. Local circular maxima become the reported heading hypotheses.

The heading objective adds three terms to the existing observation loss:

```text
L_heading = 0.5 L_mixture_nll + 0.1 L_circular + 0.05 L_unit_circle
```

`L_mixture_nll` trains the complete multimodal circular posterior,
`L_circular` optimizes geodesic angular distance without a +/-180 degree
boundary, and `L_unit_circle` constrains each learned direction encoding.
The model also reports an antipodal overlap score. A value near one indicates
strong theta versus theta+pi ambiguity; a value near zero indicates a single
directed mode.

Train the new large variant without changing the legacy PersistentBearing run:

```bash
DEVICE_ID=0 EPOCHS=100 BATCH_SIZE=20 GRAD_ACCUM_STEPS=5 NUM_WORKERS=8 \
bash scripts/persistent_train_sac_large.sh
```

The old DINO cost-volume variant remains available with:

```bash
HEADING_DISTRIBUTION=legacy bash scripts/persistent_train_large.sh
```

Both variants use the same test command. SAC checkpoints additionally write
heading median/P90/P95/max errors, 45/90-degree failure rates, symmetry,
entropy, concentration, heading confidence ECE, and the top circular modes to
`metrics.json` and `predictions.csv`.

The cached weight is detected automatically. It can also be specified explicitly:

```bash
export DINOV3_WEIGHTS=/path/to/dinov3_convnext_small_pretrain_lvd1689m.pth
```

## Smoke Training

Run one batch on the small Singapore dataset:

```bash
DATASET_DIR=../Bearing_UAV_90K/c1_254k_37bc_b15_s1_v3d \
EPOCHS=1 BATCH_SIZE=2 NUM_WORKERS=0 \
MAX_TRAIN_BATCHES=1 MAX_VAL_BATCHES=1 FOREGROUND=1 \
bash scripts/persistent_train.sh
```

## Full Training

Small model:

```bash
DEVICE_ID=0 EPOCHS=20 BATCH_SIZE=16 NUM_WORKERS=8 \
bash scripts/persistent_train.sh
```

Large model:

```bash
DEVICE_ID=0 EPOCHS=30 BATCH_SIZE=20 GRAD_ACCUM_STEPS=5 NUM_WORKERS=8 \
bash scripts/persistent_train_large.sh
```

The large default uses a frozen DINOv3 ConvNeXt-Base, 256-dimensional
dense features, three domain-specific MLP blocks per branch, and three
shared metric MLP blocks. Its 8,675,331 trainable parameters are separate
from the 88,591,464 frozen foundation-model parameters. A micro-batch of
20 with five-step gradient accumulation gives an effective batch size of
100 without exceeding the 24 GB GPU memory.

Follow progress:

```bash
tail -f log/persistent_bearing/train_*.log
```

## Independent Test

```bash
FOREGROUND=1 bash scripts/persistent_test.sh \
  results/persistent_bearing/<run>/best_model.pth
```

Do not apply online memory to the randomly ordered 9k independent test split.
Reset it for every independent sample, and reset it once at the start of each
ordered navigation route.

## Continuous Navigation

The persistent runner is separate from the baseline navigation program:

```bash
FOREGROUND=1 RSI_ID=37bc TRAJ_ID=50 \
bash scripts/persistent_nav.sh \
  results/persistent_bearing/<run>/best_model.pth
```

Validate model loading without starting the route:

```bash
DRY_RUN=1 FOREGROUND=1 bash scripts/persistent_nav.sh \
  results/persistent_bearing/<run>/best_model.pth
```

For every route, the runner creates a new memory instance. Local top-K
`(x,y,theta)` candidates are converted to a route-level map coordinate system
before Bayesian propagation and memory updates. Only the final posterior is
allowed to update tentative/confirmed memory.
