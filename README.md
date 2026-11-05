# OUEM Canopy Segmentation

Engineering repository for a practical **RGB tree-canopy semantic-segmentation component** of the **Open Urban Environment Model (OUEM)**.

**Goal:** Use Japanese GSI aerial RGB imagery at approximately 0.5–0.6 m/ground pixel to produce usable 2D tree-canopy masks for OUEM's simplified 3D canopy, shade/heat and landscape workflows. The priority is a maintainable, reproducible implementation with documented rights and limitations—not a state-of-the-art benchmark or individual-tree delineation.

## Active route and current status

**Active engineering route (2026-11-05):** initialize the official Ai2 SatlasPretrain `Aerial_SwinB_SI` backbone and FPN, attach a newly initialized two-class semantic segmentation head (background versus Tree/Canopy), and fine-tune on **verified CC BY 4.0 OAM-TCD** RGB imagery and labels. The current preprocessing target is approximately **0.50 m/local ground pixel**, using ground-scale-aware resampling with a shared image/label grid. Training with other datasets and target-domain adaptation are deferred; no new Japanese annotation is required for the first milestone.

The route has progressed beyond the original design proposal, but **M1 fine-tuning on a useful dataset and an independently validated GSI model are not yet complete**.

| Stage | Observed status | Evidence |
| --- | --- | --- |
| M1-1 official Satlas model/weights on Windows CPU | Official checkpoint loading and CPU structural smoke verified | [M1-1 specification and records](docs/implementation/M1_1_SATLAS_OFFICIAL_CPU_SMOKE_SPEC.md) |
| M1-2 OAM-TCD source QA and local-ground-scale preparation | Two approved pairs prepared at ~0.50 m ground GSD; per-source rights and geometry checked | [M1-2D pair result](docs/implementation/M1_2D_PAIR_RESULT.json), [geometry/preprocessing note](docs/implementation/M1_2D_GROUND_SCALE_PAIRS.md) |
| M1-3 real-data CPU forward/backward and resource audit | Completed for the two selected samples | [M1-3A](docs/implementation/M1_3A_REAL_CPU_SMOKE.md), [M1-3B](docs/implementation/M1_3B_TRAINING_RESOURCE_AUDIT.md) |
| M1-4 bounded CPU learning | 20 actual FP32/AdamW optimizer steps completed on one Train crop; separate-process checkpoint reload verified | [M1-4 record](docs/implementation/M1_4_BOUNDED_OVERFIT.md) |
| M1-4B deterministic CPU continuation | Reference versus resumed step 21 matched under the recorded CPU/software conditions | [M1-4B record](docs/implementation/M1_4B_CONTINUATION.md) |
| M1-5A dataset inventory | 4,608 rows inventoried at fixed OAM-TCD revision; expansion candidates remain provisional, not training-approved | [M1-5A inventory](docs/implementation/M1_5A_DATASET_INVENTORY.md) |
| Next: M1-5B / GPU smoke | New-pair acquisition/QA and real Satlas CUDA training still pending | Plan and implement as separate, bounded changes |

**Evidence boundary:** The 20-step CPU experiment uses **Train image 661** (top-left 256 × 256 crop) and evaluates **Val image 1972** (top-left 224 × 224 crop). Improvement on these two crops demonstrates trainability, **not generalization or Japanese/GSI accuracy**. GPU VRAM/real-model training and production/commercial deployment are not established by these CPU results.

## Where to find the authoritative information

- **Requirements and responsibilities:** [Project Charter](docs/PROJECT_CHARTER.md), [Design Principles](docs/DESIGN_PRINCIPLES.md), [Satlas-first ADR](docs/decisions/ADR_20261101_SATLAS_FIRST.md), [OUEM handoff](docs/OUEM_HANDOFF.md).
- **Executable code:** `scripts/` (one bounded operation per stage); corresponding offline/unit checks in `tests/`.
- **Implementation results:** `docs/implementation/` (stage-specific methods, inputs, hashes, observed results and limits).
- **Environment:** [Windows CPU setup](docs/development/WINDOWS_CPU_SETUP.md) and its [observed package snapshot](docs/development/M1_1_CPU_ENVIRONMENT.txt). The setup document records **M1-1-era observations**; its historical “pending” entries should be read alongside the later M1-2 through M1-5A results above.
- **Historical research and license analysis:** `docs/research/`. The [earlier experiment plan](docs/EXPERIMENT_PLAN.md) is **superseded** where it conflicts with the active charter/ADR.

The charter and design documents retain their original decision-stage wording as part of their provenance. This README reports **current execution status**; stage reports are the authority for what was actually tested. Do not infer a PASS in a newer environment from a historical result.

## Existing code and reproducibility

The core work is **already scripted**, but there is **not yet one general end-to-end download/prepare/train command**. Earlier scripts deliberately target bounded inputs; avoid treating them as unrestricted pipelines for all 4,608 images.

| Operation | Existing implementation | Notes |
| --- | --- | --- |
| Bounded original-data acquisition and label QA | `scripts/m1_2a_data_qa.py`, `scripts/m1_2b_overlay_qa.py` | Initial two records; enforce rights and label checks |
| Geospatial geometry and ~0.50 m data-pair generation | `scripts/m1_2d_geometry_audit.py`, `scripts/m1_2d_prepare_pairs.py` | Current pair preparation is limited to approved IDs 661/1972 |
| Verified training-patch loading | `scripts/m1_3a_loader.py` | Uses exact relative paths, hashes and grid/validity checks |
| Real-data CPU model checks and resource audit | `scripts/m1_3a_real_cpu_smoke.py`, `scripts/m1_3b_resource_audit.py` | Read stage records for invocation and limits |
| Bounded optimizer updates and continuation | `scripts/m1_4_bounded_overfit.py`, `scripts/m1_4b_continuation.py` | CPU experiments, not general training CLI |
| Dataset-wide metadata inventory | `scripts/m1_5a_inventory.py` | Metadata only; does not authorize or acquire all new image pixels |

**For a fresh Windows checkout:**

1. Follow [Windows CPU setup](docs/development/WINDOWS_CPU_SETUP.md) and reproduce or record the resolved Python/packages. The observed baseline was Python 3.11.9, PyTorch 2.7.1+cpu and `satlaspretrain-models` 0.3.1; other machines/configurations must be checked rather than assumed identical.
2. From the repository root, run the non-GPU regression tests:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest -q
   ```

3. Obtain the **source data, processed data, official initializer and trained checkpoints separately**. They are not committed to Git. Use an external data/model cache, not the repository or its virtual environment. Preserve the manifest and relative paths such as `train/661/rgb.tif` and `val/1972/canopy_mask.tif`.
4. Use the exact CLI and acceptance conditions in the linked **stage document** before rerunning an expensive, destructive or networked operation. Many stages require a **fresh output directory** and fixed source revision/hashes. Do not overwrite accepted artifacts.
5. Compare expected and observed hashes; where software-dependent GeoTIFF byte-for-byte equality is not guaranteed, also check pixel content, CRS, affine transform, extent and label semantics. Keep source and attribution records with the data.

The two existing processed pairs are documented in [M1-2D result](docs/implementation/M1_2D_PAIR_RESULT.json): `train/661` is 264 × 264 and `val/1972` is 231 × 231 before patch cropping. The fixed OAM-TCD revision is `d97d4da0ebbb6e249ae95ac5e19656babd972eb2`. Existing files are external to the repository; a new PC must receive or regenerate them and verify the hashes.

**Next engineering rule:** For M1-5B, extend the existing acquisition/QA/preprocessing logic to small additional source groups with configurable paths, network limits, stable manifests and separate offline processing. Prefer reusing existing scripts/tests and updating existing documents over creating a parallel pipeline or numerous new files. Windows/CPU portability and non-interactive operation matter for eventual company-PC use; Google Colab GPU execution is a distinct environment-specific gate.

## License, evaluation and deployment cautions

Ai2 lists SatlasPretrain weights under **ODC-BY** and the model-loading library under **Apache-2.0**. The upstream ImageNet initialization chain and eventual internal/commercial deployment obligations remain open for review; these facts are **not a final legal clearance**. Use only OAM-TCD training records with verified **CC BY 4.0** dataset/annotation and original imagery-source rights and attribution. Do **not** use Restor's released noncommercial/restricted pretrained model weights in this training chain.

Keep official OAM-TCD holdout and independent Japanese references out of training and development tuning. GSI qualitative inspection does not establish quantitative target-domain accuracy; GT54's existing land-cover classes are not automatically canopy ground truth. No public trained-model/checkpoint redistribution is planned.

## Project history

This repository was formerly `oam-tcd-poc`. Prior scale/edge-effect experiments, older A/B/C adaptation plans and superseded model routes remain in history and research records, not in the current implementation mandate.
