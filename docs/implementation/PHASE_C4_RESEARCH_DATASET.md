# Phase C4 fixed research Train/Val candidate

Issue #75 starts from main `bd05540160d74617d757e1cb6e49f2d3c3ac0205`.
The external output is `phase-c4-dataset-20261114-01` under the audit root.
The frozen candidate contains **13 Train sources / 26 windows and 7 Val sources /
7 fixed windows**. Its source technical counts are 11 PASS / 9 HOLD. These
retain their historical technical and Formal R1 statuses. AI screening and
independent user review are distinguished in the review record.

## Data selection and results

The nine C3 CLEAR sources and fifteen products sets were rechecked and reused.
Additional source selection used Phase B geography, original teacher fraction
and technical status before inspecting any model output. Thirteen source-specific
official STAC GETs returned HTTP 200, totaling 54,774 body bytes. Dataset/source
identity, CC BY 4.0, named producer/licensor and attribution were checked with
the existing function. 824 lacks licensor attribution and remains HOLD without
product generation. Twelve additional sources produced nineteen windows.

3986's sampled window has sharp internal black holes (10 exact-black pixels,
65 pixels with all channels <=5) within dark forest. Coverage versus shadow or
encoding remains unresolved. This window is withheld without a NoData claim or
new color threshold. The previous two 1054 label-ambiguity windows and two 4315
white-margin windows remain withheld. Other positions in those images remain
unreviewed. Of the 110 geometric candidates, 86 sources remain unexamined across
C3/C4. No candidate was substituted automatically.

| Fixed Val ID | Crop xywh | Original / patch canopy % | Source center lat/lon | Landscape observed |
|---|---|---|---|---|
|1241|162,99,224,224|0.97 / 0.58|-15.842,-49.261|Brazil, sparse pasture crowns|
|205|80,111,224,224|29.30 / 21.47|18.527,-88.310|Belize, scattered trees and wooded edge|
|461|18,108,224,224|11.52 / 0|34.708,135.311|Japan, sandy shoreline/water|
|1575|83,158,224,224|87.19 / 93.37|-16.319,168.308|Vanuatu, dense palms/forest|
|1137|40,116,224,224|53.77 / 49.32|-4.736,55.521|Seychelles, residential greenery|
|1609|78,38,224,224|58.42 / 57.04|27.071,86.011|Nepal, hillside/road and tree cover|
|4328|9,49,224,224|8.85 / 11.94|45.239,5.646|France, agriculture and bare/leafless crowns|

Countries are broad geographic interpretations of the recorded coordinates.
RGB, teacher and overlay inspection found broad correspondence for these
windows. Fine annotation completeness remains NOT_VERIFIED. Val positions,
sources and seed 0 are pinned before model evaluation. No cover quota was
optimized. The seven patch fractions span 0–0.9337, median 0.2147.

Train IDs are 79, 171, 596, 717, 1306, 2493, 3179, 3206, 3257, 3367, 4038,
4534 and 4601. Each has two distinct seed-0/1 positions. They cover sparse dry
vegetation, water/docks, planted urban trees, residential/agricultural scenes
and forest. Patch fractions span 0.00020–1, median 0.1221; two are below 1%,
two above 99%, and one is exactly all-one. Uniform teachers are preserved.
Within-source windows overlap and are correlated. A 200-step proposed schedule
selects each source 15 or 16 times through seeded complete source cycles.

## Fixed identity and read-only handoff

`dataset/` contains 99 byte-copied GeoTIFFs, a v2 crop plan and six bounded preview
pages. Product SHA matched before/after copying; the existing reader rechecked
all products, grids and saved-fraction thresholds. Forty original TIFF/PNG SHA
pins matched. Metadata checks found unique source groups and no cross-role
object aliases/footprint conflicts. Existing separation checks against the
official Train/Test metadata catalog gave a minimum conservative distance
lower bound of 2,282 m. This is finite metadata evidence, with unknown full-scene
coverage and aliases outside the catalog left unverified. Test pixels were unused.

| Artifact | SHA-256 |
|---|---|
|dataset/crop_plan.json|5d3dc4874baea61b5d549bd333dfb3172d7f2d93b143cae0bdbf6d459511a399|
|research_review.json|a683f89726f816273e96af65c6451fd85e00f678dae42561f4b80806522c3b4f|
|dataset_freeze.json|b4d38fa6e5c9a2be90cb9748819d1af4f537660c80ec6ec3889bac101a87f82c|
|gpu_execution_plan.json|eb38e23103d1760a2f2dd41a205d429c324293585076e9b8beee18d035b61eb4|

Dataset ID is `ouem-research-5d3dc4874baea61b5d549bd333dfb3172d7f2d93b143cae0bdbf6d459511a399`.
The complete identity also pins candidate and review SHA. Its canonical digest
is `d0a13a402ce472e6c6321db98855b22a866ebc3169f88f040b8bbb7b2ed0bc31`.
The freeze status is FROZEN_RESEARCH_CANDIDATE_PENDING_INDEPENDENT_REVIEW.
C3 schema/purpose is retained as the compatible reader contract, with
`research_phase=C4` and parent plan/review hashes recording this dataset lineage.

```python
from research_data import ResearchData
from r2_train import make_plan
data = ResearchData(
    '<C4_ROOT>/dataset',
    '5d3dc4874baea61b5d549bd333dfb3172d7f2d93b143cae0bdbf6d459511a399',
    '<C4_ROOT>/research_review.json',
    'a683f89726f816273e96af65c6451fd85e00f678dae42561f4b80806522c3b4f',
    candidates='<C4_ROOT>/candidates.json',
    candidates_sha256='bf71faf1ab512a326804f808aeff541f8e1363c8dece6d2250c6ebe7cbece0db')
plan = make_plan(data, seed=0, steps=200)  # Proposed schedule only.
rgb, target, products = data.patch(plan['val'][0], 'val')
```

## Resource observations and small code change

New-source generation took 19.96 s with process peak Working Set 263,753,728
bytes. Dataset assembly took 1.37 s; reader preflight 0.63 s; 52 Train + 7 Val
tensor supplies 1.10 s. This visits each Train source four times. The assembly/
reader process peak was 256,937,984 bytes including Python/PyTorch imports and
catalog checks. Timings are single local observations concurrent with CPU tests.
The final dataset has 106 files / 9,750,137 bytes. C4 output including intermediate
products is about 15.8 MB. A simple proportional estimate for 201 windows
(91 Train x2 + 19 Val) is about 60 MB, excluding additional evidence. It is an
estimate, with rights/quality and actual runtime still unmeasured for the remainder.

`research_patches.py` now copies each RGB preview crop and flushes a contact sheet
every six windows. Pages record paths, hashes and row counts. It retains only
one page's copied crops and overlays. C1's single preview
layout is preserved. Regression tests exercise both one-page and 6+1-page output,
source resampling once and existing tiny CPU training/continuation. Final focused
tests passed 67; full pytest passed **612, with 1 skipped**, 4,842 warnings,
226.12 seconds, exit 0. Existing R1a/Trainer implementation is preserved.

## Five prioritized tasks before the first research GPU run

1. Independently review the frozen research identity, AI quality observations,
   held windows and intended use. Approve the new run's device, step budget and
   fresh local/Drive destinations. Formal R1 remains HOLD.
2. Add a small research CUDA entrypoint using ResearchData/make_plan, the existing
   G2 CUDA timing/finite-gradient/state helpers and a dataset-bound checkpoint
   kind. The current r2_train.train core enforces CPU FP32; the existing G2 CLI
   is fixed to Zenodo10. The GPU connection needs a focused implementation PR.
3. User selects the GPU runtime and mounts Drive. Check existing CUDA PyTorch,
   torchvision, satlaspretrain-models==0.3.1, numpy, rasterio, pyproj and Pillow.
   Install missing dependencies selectively. Treat pip check as diagnostic.
   Copy/verify the fixed dataset and obtain the official Aerial_SwinB_SI
   initializer from the existing pinned HF revision
   `b1f8de04dacf9f9ed9640fe32484947710b47ede`; expected SHA is
   `f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036`
   and size 504,439,674 bytes. Current GPU setup and downloads are unexecuted.
4. Proposed first run is 200 steps, 224px/batch1/FP32, seed0, AdamW lr1e-4 and
   weight_decay0.01, all parameters trainable, augmentation/scheduler/AMP/TF32 off.
   Reuse the official model factory with pretrained SWINB/FPN and fresh official
   Upsample/SEGMENT 2-class Head, RGB float32/255 and long 0/1 targets, official CE.
   Save at steps100/200 with model/optimizer, CPU/CUDA/Python/NumPy RNG, schedule,
   identity, initializer SHA, Git and environment. Verify persistent-copy SHA and
   restoration. Log loss, memory, timing and finite updates; stop on OOM/NaN.
   These are proposed settings requiring execution approval. GPU continuity and
   numerical reproducibility remain unverified.
5. Evaluate reconstructed initial/final models on exactly the seven frozen Val
   windows with eval/inference_mode and argmax/softmax. Reuse G3 count_metrics,
   confusion and aggregate for canopy IoU/F1, precision/recall and CE, retaining
   macro/micro and null undefined metrics. Inspect RGB/teacher/prediction/FP-FN/
   probability images, then decide further training. Retain fresh-head baseline
   reconstruction limits and avoid GSI/generalization claims from this subset.

The present work performed tensor supply and synthetic-model tests only.
Real-model training, GPU execution and initializer acquisition remain pending.
