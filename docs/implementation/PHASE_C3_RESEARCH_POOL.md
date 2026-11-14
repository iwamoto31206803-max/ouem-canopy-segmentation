# Phase C3 source-balanced research pool

Issue #73 starts from main `08c1cdbc8345d5e848dc8620e155bc42c3c50aae`, which
contains #67/#70/#72. The existing Phase B SHA is
`6643f2b4037cc967eeac63d2cb2460bc0c2107f568c47a311d9f12b849e07867`.
Its 150 records contain 110 geometric candidates (91 Train / 19 Val), 39
impossible images and one RGB-format failure. Candidate technical counts are
52 PASS / 58 HOLD. Formal R1 remains HOLD.

## Small reusable extension

`research_patches.py --build-candidates` projects Phase B and local official
metadata into `ouem-research-candidates-v1`. It records source/fold/split, SHA,
metadata reference, shape, placements, original teacher fraction and unresolved
rights/patch review. It checks official Train membership and unique source
groups using the existing local index. Source pixels are read only for explicit
`--ids` during generation. The CLI retains its C1 behavior when no candidate
manifest is supplied.

Generation with pinned candidates emits `ouem-exploratory-crops-v2`, keyed by
image ID and full-grid crop coordinates. `--train-seeds 0 1` creates at most two
distinct Train windows per source; `--seed 0` fixes one Val window. Coordinates
are selected directly on the bounds-preserving approximately 0.50m grid,
avoiding positive-area support of edge-connected exact black. The original
R1c-0/Phase B algorithm, RGB area mean, float32 fraction and >=0.5 threshold are
reused. White support remains diagnostic. RGB/fraction full-grid arrays are
computed once per source. Only the small window set is written. Duplicate
seed positions are deduplicated. Output directories are exclusively created.

`ResearchData` accepts the versioned manifest and
`ouem-research-patch-review-v2`. Reviews contain `image_id`, `crop_xywh`,
`decision` and `reason` for every generated window, plus reviewer, plan SHA and
Formal R1 HOLD. Only CLEAR_FOR_EXPLORATION windows are exposed. Candidate,
plan, review and product hashes bind the input identity. C1/C2 v1 remains
supported. Source identity/rights and technical product checks are reused.
Historical HTTP metadata identifiers are accepted in v2 only for the pinned
official OIN host/object path. They remain identifiers, without HTTP retrieval.
This was needed for 4038, whose original metadata retains an HTTP spelling.

The reader exposes a small `crop_positions` index. The existing Trainer shuffles
source IDs in complete cycles and then chooses one reviewed position per
selected source using its seeded RNG. A source with more possible windows gains
no extra selection weight. Exactly one reviewed Val position per source is
required. The schedule and candidate/review identity persist in checkpoints.
The data path reads three stored 224px products per sample, with SHA checks;
2048px source analysis is performed during bounded generation. This limited
precomputed pool can be refreshed through another explicit seed plan. All
stride1 windows are never written out.

## Reproduction

Use a fresh external output root and the existing source-bound STAC directory.
The first command reads metadata only. Review the generated candidate SHA before
supplying it to generation. The eleven IDs below are the measured pilot scope.

```powershell
.\.venv\Scripts\python.exe -B scripts/research_patches.py `
  --root '<ZENODO_ROOT>' --diagnosis '<PHASE_B>/patch150_feasibility.json' `
  --diagnosis-sha256 6643f2b4037cc967eeac63d2cb2460bc0c2107f568c47a311d9f12b849e07867 `
  --build-candidates --output '<NEW_ROOT>/candidates.json'
.\.venv\Scripts\python.exe -B scripts/research_patches.py `
  --root '<ZENODO_ROOT>' --diagnosis '<PHASE_B>/patch150_feasibility.json' `
  --diagnosis-sha256 6643f2b4037cc967eeac63d2cb2460bc0c2107f568c47a311d9f12b849e07867 `
  --candidates '<NEW_ROOT>/candidates.json' `
  --candidates-sha256 bf71faf1ab512a326804f808aeff541f8e1363c8dece6d2250c6ebe7cbece0db `
  --evidence '<EXISTING_STAC_COPY>' --output '<NEW_ROOT>/pairs' `
  --ids 1054 2493 3206 4038 4315 596 717 171 1241 205 461 --train-seeds 0 1 --seed 0
```

After recording the actual window review and its hash, supply it to the reader.

```python
from research_data import ResearchData
from r2_train import make_plan
data = ResearchData(
    '<NEW_ROOT>/pairs',
    '61ed551059dabe08e8966c20549f18a9ecdc44a95986a9a5bbba7dc92dce47e8',
    '<NEW_ROOT>/research_review.json',
    '2f05a66c1724eb9b1fa91e63a02d02257d79bfef0b03429bbe9dfd662aad7d4b',
    candidates='<NEW_ROOT>/candidates.json',
    candidates_sha256='bf71faf1ab512a326804f808aeff541f8e1363c8dece6d2250c6ebe7cbece0db')
plan = make_plan(data, seed=0, steps=30)
image, target, products = data.patch(plan['val'][0], 'val')
```

## Local pilot observations

External output is `phase-c3-110-supply-20261114-01` under the local audit root.
Existing individual STAC evidence was reused for 11 sources, with zero network
requests. Selection used existing evidence availability and different roles,
technical states and teacher fractions. Model outputs were not used.

| Source | Role | Generated windows | Binary teacher fraction | AI research screening |
|---|---|---|---|---|
|1054|Train|2|0 / 0|Both LABEL_REVIEW_REQUIRED|
|2493|Train|2|0.12498 / 0.08275|CLEAR_FOR_EXPLORATION|
|3206|Train|2|0.64886 / 0.68292|CLEAR_FOR_EXPLORATION|
|4038|Train|2|0.04309 / 0.04448|CLEAR; white roofs retained|
|4315|Train|2|0.13343 / 0.22887|Both COVERAGE_REVIEW_REQUIRED|
|596|Train|2|0.00020 / 0.00030|CLEAR; water/docks, tiny positives retained|
|717|Train|2|0.11916 / 0.09082|CLEAR; crown rows in dry fields|
|171|Train|2|1 / 0.99982|CLEAR; dense forest, uniform teacher retained|
|1241|Val|1|0.00584|CLEAR; sparse pasture crowns|
|205|Val|1|0.21468|CLEAR; scattered crowns/wooded edge|
|461|Val|1|0|CLEAR; sandy shoreline, zero teacher retained|

There are 8 Train / 3 Val sources, including 5 technical PASS / 6 HOLD. Nineteen
windows and 57 TIFFs passed readback, threshold, grid and SHA checks. All selected
black-support counts are zero. 4315 has diagonal white artificial margins in both
windows (35,673 / 24,645 white-support cells). 4038's first window has 328 such
cells along bright roofs, consistent with previous evidence. No color-to-NoData
rule was applied. The two 1054 windows retain unresolved vegetation/zero-label
semantics. Only those sampled windows are withheld; its other positions remain
unreviewed candidates. Fine label completeness remains NOT_VERIFIED throughout.

AI screening leaves 15 candidate windows from 6 Train / 3 Val sources. Human
independent review and final training selection remain pending. Thirty planned
Train samples use each of the six sources five times. Seed changes preserve
the three Val windows. Real-model optimizer steps are zero.

Generation took 17.72 seconds, peak process Working Set 266,006,528 bytes.
Products plus plan/CSV/preview total 5,072,690 bytes. Reader initialization took
0.367 seconds; supplying 30 Train + 3 Val tensors took 0.616 seconds, with peak
process Working Set 237,830,144 bytes including Python/PyTorch imports. These
single-run observations used local files and coincided with CPU regressions.
Contact sheet construction retains preview arrays; memory at 110 sources remains
unmeasured and should be checked when that larger scope is authorized.
The 22 source pins and 57 output hashes matched independently. All five C1
seed-0 product sets match their original TIFF hashes. Existing C1/C2 products,
reviews and source data retain their identities.

## Expansion conditions

Individual source rights were checked for these 11; the remaining 99 candidates
are NOT_VERIFIED in this work. Resolve only the selected sources' missing evidence
and review their finite window pools before exposing them to training. Keep
ambiguous windows withheld and retain Formal R1 HOLD.

Phase B global teacher fractions span 0–1 for Train (median 0.1421) and
0–0.9149 for Val (median 0.1152); 18/19 Val sources contain some original positive
labels. A chosen crop can still be empty. Current fixed Val fractions are
0/0.00584/0.21468, so two windows are background-heavy. Extend fixed Val coverage
before performance claims, using geometry/landscape/teacher evidence before
model evaluation. No cover quota was optimized. Review this implementation and
pool before authorizing a larger generation scope or GPU connection.

Tests exercise variable source counts, balanced cycles, reproducible positions,
fixed Val, window-level holds, legacy HTTP identifiers, source/role/SHA failures,
resampling once per source and synthetic CPU continuation. C1/C2/R1a regressions
remain part of the test run. Formal R1 contract and model/loss semantics retain
their existing implementation.

Final focused integration passed 122 tests with one skipped. Final full pytest
passed 611 tests with one skipped and 4,810 warnings in 222.47 seconds, exit 0.
Use a fresh short external `--basetemp` on Windows. `git diff --check` passed.
