# MyoPS380: fixed 304 train / 76 test, final checkpoint

Protocol ID: `myops380_train304_test76_last_v1`. This is a different training
protocol from the existing 243/61/76 development split. Do not mix their results.

## Data contract

- The versioned manifests in `preprocessing/splits/myops380_no_val` contain all
  1,972 training slices from 304 patients and the original 76 test patients.
- They equal I_MMSeg's train/test ID sets. The reference is stored inside SCAR;
  I_MMSeg does not need to be installed. Training seed never changes the split.
- All three modalities must have identical file IDs, aligned shapes, finite
  normalized images, and the same target mask. A missing patient or slice fails.
- Use `legacy` for the original metadata-free cache (2=scar, 3=edema). Use
  `canonical` for newly packaged SCAR caches. The loader converts once to
  0=background, 1=normal myocardium, 2=edema, 3=scar.
- Existing NPZ/HDF5 files and any old validation-volume cache are not rewritten.
  Old validation volumes are unused in this protocol. Effective train/test
  manifests, source preprocessing metadata and cache hashes are recorded separately.

## Verify before training

Run from the repository root:

```powershell
python preprocessing/verify.py --no-val --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --list-dir preprocessing/splits/myops380_no_val --label-order legacy
```

This reads every train slice and test volume to check the input contract; it does
not train, run test inference, or select a model. The report includes patient and
sample counts, canonical class voxel counts and a byte fingerprint of the cache.

`build_splits.py --no-val` can reconstruct the same manifests from a complete
cache into a new list directory. It refuses to reinterpret a directory containing
validation manifests. Raw packaging also supports `--no-val`; it requires the
complete cohort and the fixed test list, and checks the released slice inventory.
It can reuse existing valid no-val manifests read-only, including the versioned
directory used by the paper config; it does not rewrite these files.

## Freeze the paper experiment

Choose the architecture, initialization, loss, sampler, augmentation, input size,
microbatch, accumulation, LR, total epochs and seed list before final runs. The
provided experiment YAML uses current M2-Plus defaults and 300 epochs; these are
explicit starting settings, not a claim that 300 epochs is optimal.

Initialization is random unless `--pretrained` or `--init-weights` is supplied.
Record and use the intended initialization consistently across comparable runs.
Initialization files are fingerprinted. Starting from a previously trained model
must be disclosed; `--init-weights` starts a new optimizer and schedule.
For `run_all.py`, configure `train.pretrained` or `train.init_weights` in the YAML;
the corresponding CLI flags are available on `train.py`. The Colab notebook
continues to request the encoder pretrained file explicitly, as it did before.

Train and automatically test after completion:

```powershell
python run_all.py --config training/config/experiments/myops380_no_val.yaml --skip-cache --run-id m2_plus_no_val_seed1234 --seed 1234
```

The YAML already enables `train.no_val: true`. For another model config, supply
`--no-val --list-dir preprocessing/splits/myops380_no_val` explicitly. Only
MyoPS380 supports this flag. `patience` and `min_delta` must be zero; validation
fraction is ignored. Requesting `val_vol` fails. Without no-val, development
training and other dataset profiles retain their previous behavior.

Adapt microbatch/accumulation to GPU memory *before* the final run. Effective
batch size alone does not make different microbatches numerically equivalent
when BatchNorm or stochastic operations are present. The local RTX 3050 4 GiB
was checked with microbatch=1 and accumulation=2 for two real training slices;
that smoke check does not establish a suitable full-run batch size or accuracy.

## Checkpoint and resume

- `last.pt` is atomically replaced after each completed epoch. No `best.pt`,
  validation score or patience state is saved in a no-val run.
- Full model/optimizer/scheduler/scaler/RNG/loader-generator state is retained.
- Planned/completed epochs, successful/skipped optimizer updates, training policy,
  code/config fingerprints, initialization and data/split hashes are saved.
- Evaluation requires `last.pt`, the complete planned epoch budget, consistent
  update/scheduler state and exactly the locked test cohort. A 100/300 checkpoint
  is resumable but cannot be evaluated by this paper-protocol evaluator.
- Resume cannot change no-val mode, split, budget, optimizer/loss settings or
  tracked source/config contents. Moving a repository is allowed if content is
  identical; data-root follows the existing fixed-path resume rule.
- Old validation runs cannot become no-val runs through `--resume`.

Example of splitting a run across sessions (replace paths/settings as needed):

```powershell
python train.py --config training/config/experiments/myops380_no_val.yaml --run-id m2_plus_no_val_seed1234 --epochs-per-run 50
python train.py --config training/config/experiments/myops380_no_val.yaml --resume outputs/myops380_no_val/runs/m2_plus_no_val_seed1234/checkpoints/last.pt
```

Reapply training-setting CLI overrides from the original run when resuming, but
replace `--pretrained` / `--init-weights` with `--resume`; these sources cannot be
combined. Resume does not automatically restore CLI settings. `epochs_per_run` pauses the run without
changing the total schedule; finish the run before test.
Initialization settings in the unchanged YAML are ignored on resume; weights,
optimizer state and initialization provenance come from the checkpoint. The
original initialization file is not needed for resume.

Evaluate the completed run separately:

```powershell
python test.py --checkpoint outputs/myops380_no_val/runs/m2_plus_no_val_seed1234/checkpoints/last.pt --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --split test_vol --batch-size 8
```

## Results to retain

Keep `config.json`, `metrics.csv`, `metrics.jsonl`, `summary.json`, saved manifests,
`last.pt`, and test `metrics.json` / `per_case.csv` together. The test report records
the checkpoint SHA256, final epoch, planned budget, full training arguments,
initialization, source fingerprints, split/cache hashes, inference and metric
protocols. Use a fresh output directory for another evaluation.

Metric formulas and inference are unchanged: per-patient aggregation, edema
exclusive {2} distinct from inclusive {2,3}, both-empty regions excluded with
defined/undefined counts, and MyoPS380 distances in **voxels**, not mm. Inference
uses one pass, resizes logits before argmax and does not filter components.

Do not select epochs, seeds, hyperparameters or architectures by these test
results. Report every predeclared seed if using multiple seeds. Baselines intended
for direct comparison should use the same cohort and final-checkpoint rule;
document epoch/update budgets, preprocessing, regions and metrics explicitly.

## Verification

```powershell
python -m pytest -q
```

`tests/test_no_val.py` exercises the full released ID inventory with synthetic
images: strict cohort/cache checks, no validation access, uninterrupted versus
resumed training (including dropout, augmentation, a partial final batch and
gradient accumulation), rejection of partial budgets, final evaluation over all
76 synthetic test patients and run_all flag propagation. The tiny model in that
test checks pipeline behavior, not segmentation quality. Existing architecture
tests and a separate real-data GPU smoke check cover model execution.
