# Phase C2 research patch input

Issue #71 starts at main `2735f40e62f9f9085431656119d889ad6280780c`, containing
PRs #67 and #70. The merged-main focused regressions passed 87 tests with one
existing skip before adapter changes (31 C1/Trainer and 56 R1a tests).

## Five-patch screening

The C1 crop plan, ten original SHA pins and fifteen product hashes matched.
Only existing RGB/annotation/products were read. The contact sheet and a
nearest-neighbor enlarged 1054 RGB were visually inspected by Codex. These
observations are AI screening for bounded exploration; independent review and
final training selection remain pending. Formal R1 remains HOLD.

| Image | Role | Binary crown fraction | Research candidate decision |
|---|---|---|---|
|1054|Train|0.000000|EXCLUDE_FROM_FIRST_TRAINING; LABEL_REVIEW_REQUIRED|
|2493|Train|0.124980|CLEAR_FOR_EXPLORATION|
|3206|Train|0.648856|CLEAR_FOR_EXPLORATION|
|1241|Val|0.005839|CLEAR_FOR_EXPLORATION|
|205|Val|0.214684|CLEAR_FOR_EXPLORATION|

1054's original PNG contains 507,594 positive pixels overall. Its selected crop's
positive-area source support rectangle is `[277,256,1451,1430)` in original pixel
coordinates, with zero positive annotation pixels. The saved fraction and mask
are both zero. This establishes consistency with the input annotation; the
zero result did not arise from thresholding away fractional positives.
The local COCO vocabulary is category 1 canopy / 2 tree, and the full 1054 tile
has eight COCO annotations. COCO annotation IDs are not a Panoptic crosswalk.
Visible watercourse vegetation could include herbs, shrubs or trees. Its height,
category inclusion and any canopy omission remain NOT_VERIFIED. Keep this crop
excluded until a specific label review resolves the ambiguity. No label edits
or all-zero teacher workaround were applied.

2493's tree groups and 3206's wooded area broadly align with the teachers.
1241 shows two small labeled crowns in pasture; 205 shows scattered crowns and
wooded edges. No gross RGB/teacher shift was observed. Fine omissions and crown
edges remain NOT_VERIFIED. Val cover fractions differ substantially, and two
fixed patches cannot establish balanced geography, landscape or crown coverage.
Retain both positions for future reviewed evaluation, without score-based
selection. No model predictions were used in this screening.

## Reader and Trainer boundary

`research_data.ResearchData` reads the C1 three-product schema. It requires
independently supplied expected SHA values for crop_plan.json and a separate
research review. The review schema is `ouem-research-patch-review-v1`, containing
`crop_plan_sha256`, `formal_r1="UNCHANGED_HOLD"`, a named `reviewer`, and five
`samples` with integer `image_id`, `decision` and nonempty `reason`. Decisions
are CLEAR_FOR_EXPLORATION, REVIEW_REQUIRED or EXCLUDE_FROM_FIRST_TRAINING.
Only explicit CLEAR records appear in `data.samples`. A pending/excluded source
also fails direct `patch()` calls. A review record is a research selection
policy, with its author and limitations recorded; it is not a Formal R1 approval.

The reader verifies the fixed five IDs, folds/roles, distinct oam_id, source hash
records, metadata reference, three CC BY 4.0 declarations, named producer,
attribution and source-bound STAC reference/hash. Source authenticity and legal
conditions rely on the pinned C1 evidence; the reader does not fetch or adjudicate
rights. It verifies every product's bytes, safe path, dtype/count/shape,
CRS/affine/bounds, crop translation, black-support diagnostic, numeric fraction
range and saved float32 >=0.5 threshold. Product hashes and review/plan pins are
rechecked at sample use. Uniform teachers stay unchanged. A metadata mask check
is a technical consistency check, without an assertion of photographed coverage.

The data interface is `root`, `samples`, `identity`, `patch(item, split)`.
The Trainer's overlap guard now uses `data.root`; R1Data exposes its existing
preflight root through this property. R1a preflight/load_sample requirements,
model construction, CE/AdamW, schedules and checkpoint logic are unchanged.
The research identity contains purpose, plan/review SHA and Formal R1 HOLD,
and is carried into plans and checkpoints. No R1a verification object or fourth
valid_fraction product is fabricated. The only allowed local crop is the stored
`[0,0,224,224]`; original full-grid crop coordinates remain in crop_plan.json.

## Local review and example

Review file is stored separately under
`phase-c2-label-review-20261114-01/research_review.json` in the external audit root.
Its SHA is `46d0771663e8fb00ba6d91a24dde7cbca5b021051203d5a22385f41c7f1205df`.
The C1 plan SHA is `def8cf915499eb92d8d44635689141d4cd8f1a5ff8f8f6524621b011ebb1c3ee`.
All five records/products passed read-only adapter checks. Explicit selection
returned Train 2493/3206 and Val 1241/205. No real-data training was performed.

```python
# Run from the repository after adding scripts to sys.path.
from research_data import ResearchData
from r2_train import make_plan

data = ResearchData(
    '<C1_PAIRS_ROOT>',
    'def8cf915499eb92d8d44635689141d4cd8f1a5ff8f8f6524621b011ebb1c3ee',
    '<C2_REVIEW_ROOT>/research_review.json',
    '46d0771663e8fb00ba6d91a24dde7cbca5b021051203d5a22385f41c7f1205df')
plan = make_plan(data, steps=1)  # Planning only.
images, target, products = data.patch(plan['val'][0], 'val')
# RGB float32 /255 [1,3,224,224], target long [1,224,224].
# train(data, plan, model_factory, fresh_output) is tested with a tiny synthetic
# CPU CE model only. Real training remains a later reviewed task.
```

## Tests and next review

Final focused integration passed 108 tests with one existing skip, including
21 new reader tests and the C1, Trainer and R1a regressions. New cases cover
unreviewed input, split/fold/path/hash/rights/source failures, fraction/threshold,
dtype/nodata/grid/crop failures, post-open mutation, uniform masks, a tiny CPU
one-step checkpoint with research identity and fixed Val input.

The final full suite passed 597 tests with 1 skipped and 4,414 warnings in
205.80 seconds, exit 0. Use a fresh short external temporary directory on
Windows, for example `pytest -q --basetemp=<NEW_EXTERNAL_TEST_TEMP>`.
The existing skip concerns unavailable symlink privileges. Source/product
hashes remained matched, and the C1 plan/preview SHA values remained unchanged.

After independent review, extend per-source seeded selection only within a
newly agreed generation scope. Use source-balanced allocation and a fixed Val
list; consider multiple Val positions before coverage claims. Avoid materializing
all 1,536,807 overlapping windows. No 110-source generation, official weights,
GPU processing, network retrieval or real-data optimizer step occurred here.
