# Second benchmark audit — 2026-10-28

Scope: I-MMSeg architecture/objective preservation; train304/test76 without
validation; data semantics; 300-epoch final-checkpoint policy; resume; inference
and shared SCAR metrics. Only train data and synthetic cases were used for model
debugging. No real test inference or accuracy-based tuning was performed.

## Defects reproduced and fixed

1. **M2-Plus label-order override could swap evaluation GT.** The shared evaluator
   checked cache bytes and split IDs but not the declared label order against
   `checkpoint.benchmark_data.source_label_order`. An explicit regression test
   first reproduced acceptance of a wrong canonical override for a legacy
   checkpoint. It now rejects that mismatch before decoding test GT.

2. **FP16 was exposed without numerical support in the original model.** GPU
   synthetic forward/backward produced non-finite main logits, NaN loss and
   non-finite gradients in FP16. Casting only the loss inputs to FP32 did not fix
   it: the original cosine attention had already overflowed. To preserve the
   model's math, FP16 is now rejected rather than silently changing that module.
   FP32 remains default; optional BF16 model autocast uses the original loss in
   FP32. Both modes passed real-train GPU optimization checks.

3. **Cache geometry/identity checks were weaker than SCAR's.** The reader had
   checked shape, values and mask equality, but ignored patient ID and physical
   metadata. It now validates patient identity, HW/HWD axis order, finite valid
   spacing/affine pairs, geometry consistency across modalities and normalization
   metadata. Regression tests cover NPZ and synthetic HDF5 cases with identical
   array shapes but incompatible affine translations.

4. **Report metadata could override computed region scores.** Metrics were
   merged before provenance. The merge order now makes freshly computed scores
   authoritative; a regression test confirms metadata cannot turn Dice 0 into 1.

5. **Checkpoint counter types were not strict.** Numeric equality could accept
   floats/bools as epoch/update/batch counts. The gate now requires actual Python
   integers and rejects malformed counters.

## Verification evidence

- `python -m pytest -q`: **32 passed** after the fixes.
- Original loss AND gradients match the archived original trainer block exactly
  on FP32 test tensors at epoch indices 0,20,21,99,100,299, including the original
  two-GPU text-anchor multiplicity adapter.
- SCAR augmentation matches the pinned implementation across multiple seeds;
  metric source remains byte-identical to its frozen SCAR reference.
- Resume matches uninterrupted execution with partial final accumulation groups
  and stochastic inputs/dropout, using both 0 and 1 loader worker.
- A tiny synthetic model and test-only loss executed the complete 300-epoch
  control flow, produced 300 log records and a valid final checkpoint, reached
  the planned LR endpoint and created no best checkpoint. This checks software
  behavior, not full I-MMSeg training convergence.
- Shared scoring covers all 76 synthetic patient IDs, empty-mask counts,
  canonical mapping, output schema and rejection paths.
- Full I-MMSeg: two AdamW updates on `case0002_slice000`, including epoch-index
  100 contrastive/fusion, succeeded on the local RTX 3050 Laptop GPU in FP32 and
  BF16, with finite losses and gradient norms.
- The actual `train_epoch` path was additionally exercised on three real train
  slices with microbatch 1 / accumulation 2, checking the final partial group.
- All 1,972 real train slices were read with the stricter reader. No real test
  volume was decoded during this audit.

## Remaining limits

There is no full I-MMSeg 300-epoch run or real test segmentation result yet.
Physical batch 16 has not been established to fit the local 4 GiB GPU. The
microbatch-1 checks validate execution only; they do not make accumulated
batch-Dice/contrastive losses equivalent to a physical batch of 16.

The external R50-ViT-B_16.npz was not supplied for full-file validation. Initializer
mapping and rejection-before-mutation are tested with synthetic encoder weights;
without `--pretrained`, initialization is explicitly random. Freeze hardware,
microbatch, initialization and all final-run settings before comparing results.

Source checks intentionally reject resume/evaluation against modified benchmark
code. Any pre-audit checkpoints require their matching source snapshot; they must
not be relabeled as runs using this corrected implementation.
