# ADR — 2026-11-01: Rebase on SatlasPretrain for an OUEM-useful canopy model

**Decision status:** Proposed for review/merge (direction confirmed in conversation; repository design acceptance pending).  
**Type:** Product engineering direction, **not** model performance, license signoff or implementation acceptance.  
**Supersedes:** Fixed original A/B/C OAM-TCD released-weight resolution-adaptation experiment as the **active project objective**.  
**Does not erase:** Earlier P0 investigations, model comparisons, OAM-TCD dataset work, evidence ledger, or code history.

## Context

Earlier design targeted an experimental answer: whether fine-tuning a released Restor canopy SegFormer on synthetic 50cm imagery recovered quality versus unfine-tuned weights. Subsequent investigation identified noncommercial/restricted released Restor checkpoints for the desired business pathway, while the user's actual goal is not a paper or a controlled ablation. They want a broadly usable 2D canopy layer for OUEM's GSI-based 3D/thermal/landscape workflows, with **practically sufficient** accuracy and **reasonably documented commercial-use rights**. A perfect benchmark and image-by-image nationwide proof are not entry requirements.

SatlasPretrain publishes an aerial RGB single-image Swin-v2 Base checkpoint trained for ~0.5–2m/px, with an Ai2 **ODC-BY checkpoint designation**. Ai2's loader is Apache-2.0 and supports pretrained FPN as the initial choice and newly initialized segmentation heads. OAM-TCD contributes specialist labels: use **CC BY 4.0 subset only**, verify actual records and annotations. Training on approximately GSI-scaled OAM-TCD images is a plausible, **not yet validated**, engineering approach.

## Decision

Prioritize **one buildable Satlas-first route**:
1. `Aerial_SwinB_SI` initialization from Ai2's published checkpoint, official backbone and pretrained FPN as the first implementation choice.
2. Newly initialized official `Head.SEGMENT` with two output classes (background and canopy), subject to verified target shapes and void handling; **Tree and Canopy become the positive class** after verified label decoding.
3. Fine-tune on documented OAM-TCD **CC BY 4.0 only**, using reproducible resolution-matched RGB/mask transforms and grouped splits.
4. Check real Japan/GSI imagery for suitability and connect to OUEM via a model-agnostic inference contract.
5. **No full Swin scratch-versus-pretrain comparison** and no research-style architecture search unless a practical blocker requires fallback.

**Distribution boundary:** The OUEM model/checkpoint is not planned for public release or redistribution. Internal fine-tuning and eventual commercial inference are the intended uses. This narrows review without eliminating license conditions. No ImageNet images are to be downloaded directly for commercial training.

## Why this is better aligned than the former route

- **Direct alignment:** The target inference imagery is aerial RGB at approximately Satlas's pretrained scale.
- **Domain specialization:** OAM-TCD's masks give the exact canopy supervision Satlas does not itself supply as an OUEM-ready product.
- **Simpler engineering decision:** Avoid proving that Satlas beats every alternative, while retaining pragmatic fallback choices.
- **Traceable rights:** Ai2 makes a positive first-party ODC-BY statement for final checkpoint weights; independently, OAM-TCD segregates CC BY data. The original Restor pretrained release weights are not needed.
- **Preserves investments:** Existing panoptic mapping investigation, 5x downsampling, split QA, GSI imagery and OUEM spatial contract remain valuable.

## Trade-offs and open risks — not papered over

| Risk | Current position | Next verification |
|---|---|---|
| Adequacy on Japan GSI | **Unknown**, not inferred from Satlas 0.5–2m/px matching | GSI qualitative diagnostics and bounded independent local canopy labels for numeric claim |
| Hardware and implementation | 88M-class Swin-v2 Base may be costly; neither CPU nor A2000 full fine-tune established | Pin loader, synthetic forward/head/target tests, GPU VRAM micro-step |
| Label source/shift | OAM-TCD nominal ~0.1m, different geography/sensors; provider CC BY-only dataset provenance known at documentation level | Pin actual rows, image/mask QA, honest downsample assumptions |
| License clarity | Ai2 checkpoint ODC-BY is a reasonable working basis; prior Satlas code path uses ImageNet initialization | Preserve original statements, hash model, attribution and dependencies, treat downstream ImageNet interpretation as unresolved until proportionate production review |
| Practical thresholds | No evidence-based thresholds yet | Define what OUEM can tolerate before final independent acceptance |
| Architecture head | Official loader supports segmentation but behavior not executed by OUEM | Verify official FPN/upsample and `Head.SEGMENT` class-index cross-entropy semantics before implementation decisions |
| GT54 | Land-cover benchmark not necessarily reliable canopy ground truth | Preserve independent final role; validate labels before quantitative use |

## Superseded versus retained decisions

**Replaced as active requirements**
- `EXPERIMENT_PLAN.md`'s A(original 10cm)/B(degraded 50cm)/C(adapted 50cm) pretrained Restor comparison.
- The premise that comparing scratch models or demonstrating a causal pretraining benefit is required before engineering selection.
- README framing of this repository primarily as research.

**Retained**
- No commercial-candidate fine-tuning from known restricted Restor release weights.
- Strict separation of **code / input checkpoint / training data / resulting checkpoint** rights.
- `oam_id` grouped splits, clean labels, no alterations to all-zero/all-one masks, provenance hashes and reproducible tests.
- 5x resampling as an investigated *initial* candidate, not an asserted exact GSI simulation.
- Tree IoU/F1, area error, boundary and difficult-class diagnostics; no accuracy claims based only on tree fraction or visual maps.
- GT54 held out; independent Japanese quantitative evaluation is optional later for numerical GSI accuracy claims, not a gate for M1 fine-tuning or M2 visual inspection. OUEM owns geospatial/CHM/3D.
- Small reviewable PRs and no unapproved merge or training.

## Treatment of pending or legacy P1-1 implementation

The prior README noted that a separate P1-1 implementation was in development under the former experimental framework. **Do not merge or treat any such work as implementation approval merely because this design PR is merged.** Inventory that PR/branch before reuse. Reuse or adapt only compatible utilities (CC BY-only record selection, `oam_id` grouping, verified panoptic Tree/Canopy conversion, spatially aligned downsampling and deterministic manifest/tests) after review. Replace or discard any workflow that assumes the noncommercial published Restor checkpoint, original A/B/C contrast, or model-specific training semantics. The new active training specification must explicitly reference the Satlas-first architecture and approved data pipeline.

## Conditions to reopen

Reconsider the foundation model if official Satlas loader proves unusable, training hardware cannot support a viable setup, rights review reveals a blocker for internal commercial inference, or actual Japan/GSI performance fails practical tolerances after one bounded tuning effort. Prefer the **smallest effective remediation** rather than immediately launching a full architecture comparison.

## Sources and authoritative cross-reference

- [Project Charter](../PROJECT_CHARTER.md)
- [Design principles](../DESIGN_PRINCIPLES.md)
- [Pretrained foundation candidates](../research/P0_PRETRAINED_FOUNDATION_WEIGHTS_CANDIDATES_v0.1.md)
- [Satlas aerial ImageNet lineage](../research/P0_SATLAS_AERIAL_INITIALIZATION_LINEAGE_v0.1.md)
- [ImageNet internal-use rights](../research/P0_IMAGENET_UPSTREAM_RIGHTS_AND_INTERNAL_USE_v0.1.md)
- [Torchvision checkpoint rights trace](../research/P0_TORCHVISION_SWINV2B_IMAGENET_WEIGHT_RIGHTS_TRACE_v0.1.md)
- [OAM-TCD integrated issue / decision review](../research/INTEGRATED_ISSUES_AND_DECISIONS_v0.1.md)
- [Ai2 Satlas published weights](https://github.com/allenai/satlaspretrain_models/blob/main/README.md)
- [Ai2 Satlas foundation documentation](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md)
