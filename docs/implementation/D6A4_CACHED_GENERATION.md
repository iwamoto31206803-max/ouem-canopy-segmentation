# D6-A4 — Cached OAM-TCD research generation

[Issue #116](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/116) authorized one bounded CPU campaign. The immutable dataset readback passed at **80 original groups (75 Train, 5 fresh Val)**. The 200-group target remains short by 120. The available local STAC evidence covered 122 distinct groups before eligibility screening. Official Train cache availability covered 3,341 training-fold and 828 fold-0 tiles. The tile counts describe files, while the group counts determine independent sampling units.

## Candidate selection and results

The common candidate roster was frozen before any new pixel QA. The metadata screen used the pinned official catalog, existing STAC source identity and rights, same-source aliases and known envelopes, the 1 km guard, and historical v5/Val18 separation. When an earlier eligible A1 result matched, its candidate was preferred within the group. The seed was 116, with seeded ordering within 15° catalog center cells and rotation across cells. New source groups ranked ahead of historical Train reuse within each role. Train and Val kept official folds. There was no model-dependent selection or per-image visual gate.

| Stage | Train | Val | Total |
| --- | ---: | ---: | ---: |
| Documentary candidates for this campaign | 79 | 5 | 84 |
| Prior eligible A1 results reused within those candidates | 14 | 3 | 17 |
| New A1 QA | 65 | 2 | 67 |
| New A1 research-eligible | 59 | 0 | 59 |
| New A1 HOLD | 6 | 2 | 8 |
| Frozen eligible production roster excluding prior pilot | 73 | 3 | 76 |
| Prior A3-2b patches reused in place | 2 | 2 | 4 |
| Integrated generation | **75** | **5** | **80** |

From the 122 cached STAC groups, 22 had no candidate tile inside the STAC bbox, 10 conflicted with historical protected identities, two had incomplete or conflicting declaration/license evidence, and four were already in the pilot. All eight pixel QA HOLDs had no 224px window clear of the existing outer-connected black support. A1 thresholds and panoptic-to-binary interpretation were unchanged. Group-level eligibility remains conditional on known-source evidence and does not establish worldwide geographic independence.

Of the 80 groups, 56 were same-role historical v5 Train source groups and 24 were new to v5 (19 Train and five Val). The group centers span 44 Train and four Val cells at the selection granularity. Two sources retained an external expected SHA and 78 had matching locally observed hashes. Local hashes support reproducibility and leave publisher authenticity unverified.

## Generation and readback

The shared eligible roster froze the pilot digest and exact shard boundaries. Each production plan and approval SHA bound its roster slice, original A1 crop and grid, current input bytes, protection references, and finite resource limits. The serial process ran one Windows Job at a time. Before proceeding, the first shard reopened all 75 TIFFs through D6Data. The next shard budgets used the first observed runtime and size with a margin and the existing per-source limits. Allocated file sizes were sampled and do not represent an OS quota.

| Stage | Sources | Wall s | Peak RSS B | Peak commit B | Logical B | Peak observed allocation B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 QA | 67 | 82.14 | 839,311,360 | 846,155,776 | — | — |
| Shard 01 | 25 | 49.46 | 802,312,192 | 780,660,736 | 3,607,636 | 3,831,664 |
| Shard 02 | 26 | 50.33 | 803,024,896 | 781,750,272 | 3,690,469 | 3,922,016 |
| Shard 03 | 25 | 48.90 | 802,181,120 | 780,697,600 | 3,521,287 | 3,749,744 |

Production CLI time totaled 148.70 s and each shard returned exit code 0. The attempt deadlines were 900 s for shard 01 and 600 s each thereafter; logical and observed allocation limits were 16/64, 19/76, and 18/72 MiB. Existing 4 MiB per-source staging, 120 s per-source watchdog, 2,560 MiB Job commit cap, 2 GiB peak RSS stop and 256 MiB free-disk floor were retained.

D6Shards verified the aggregate and roster digests, shard membership, uniqueness, crop/affine/CRS, product digests, dtypes and ranges, and the saved Float32 fraction threshold of 0.5. A CPU `make_plan` cycle visited every Train patch once and every Val patch at its fixed crop, with all tensors read through the Reader. The aggregate checks took 4.56 s. The 240 TIFFs cover 4,014,080 patch pixels and the referenced generations occupy 11,415,353 logical bytes including bookkeeping. Train mean canopy fraction was 25.63%, range 0–100%, with five zero and six positive patches below 1%. Val mean was 19.30%, range 5.63–27.58%, with neither zero nor below-1% patches.

- Executed code `a82bb2634fb90a77752533a74aea2d8b73a31d32`
- Aggregate generation `37ce54926eeda7e719dfa9f21b84939c43dfbb45d3bb328e467949ccd25299a3`
- Aggregate manifest SHA-256 `a7bd8d6eeaec24f03184e6d26253711405ad261398cd9ae0077f2845c1312c45`

## Use of the frozen generation

Keep the private aggregate manifest and its pinned roster together, with the shard directories at the bound local locations. D6Shards takes the aggregate manifest path and its independently retained SHA-256. It provides the same `samples`, `crop_positions`, `identity` and `patch(item, split)` interface as D6Data, so `r2_train.make_plan` can use it directly. On a failed shard, preserve completed immutable outputs and validate any explicitly recorded verified prefix before use. The observed run completed the full frozen roster without invoking this fallback.

Use the existing producer command from the [pilot runbook](D6A3_BOUNDED_PILOT.md) with each frozen A4 shard plan and its campaign approval pin. The four-source pilot implementation was independently reviewed, the owner authorized its scope, and private pins were reconciled locally by Codex. The same distinction applies to the private execution evidence here. Do not move or overwrite historical v5, Step280, fixed Val18, the prior pilot, or protected Test data.

A larger cohort needs usable rights/source receipts for more distinct groups and additional independent fold-0 groups. The five fresh Val groups make this an exploratory dataset with limited validation diversity. GT54 remains deferred and geographically unverified, D6-B is deferred, and Formal R1 remains HOLD for all 150. No extra downloads, training, or target-domain performance claim was made.

## Verification environment

Windows pytest passed 780 tests with two existing symlink privilege skips and zero failures. Pending deprecation notices were filtered in the full run; the duplicate ZIP member test emitted its expected warning. The D6 targeted suite passed 83 tests during integration, and the final-code aggregate integration tests passed 2/2. The documented execution commit precedes this documentation-only update.
