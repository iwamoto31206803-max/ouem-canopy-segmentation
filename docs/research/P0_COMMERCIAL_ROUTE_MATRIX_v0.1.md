# P0 Commercial-Route Matrix — Initial Source Audit (v0.1)

**Date:** 2026-10-31 | **Status:** research only, unresolved commercial compliance; no code changes or model selection. Based on main-branch integrated review v0.1. Model-selection requirement: commercially suitable, traceable construction route is mandatory.

## 1. Key finding: package-level license does not clear every encoder

The canonical segmentation_models_pytorch (SMP) repository has an overall MIT LICENSE:
- https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE
- **But its official component-specific license inventory** https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/licenses/LICENSES.md explicitly states the Mix Vision Transformer / SegFormer encoder implementation in `segmentation_models_pytorch/encoders/mix_transformer.py` is under a **noncommercial NVIDIA License**. Therefore **SMP's SegFormer encoder is not an eligible commercial-route implementation as-is**, even with randomly initialized weights. Do not mistake a package-level MIT badge for per-file clearance.
- The NVIDIA original SegFormer repo/license also restricts use of the licensed Work to noncommercial research/evaluation: https://github.com/NVlabs/SegFormer/blob/master/LICENSE .

The alternative Hugging Face Transformers SegFormer implementation `src/transformers/models/segformer/modular_segformer.py` carries an **Apache-2.0 file header** (source examined on 2026-10-31): https://github.com/huggingface/transformers/blob/main/src/transformers/models/segformer/modular_segformer.py . It may be a candidate implementation, **but the Apache header alone does not settle the independence/provenance of underlying implementation, code overlap, dependency obligations, or model-specific weights**. This requires further compliance review before clearing.

## 2. Route-level matrix

Evidence grading: **Candidate** = plausible implementation route, not commercially cleared; **Blocked** = known restriction in the identified component; **Unknown** = no verified grant.

| Route | Code / dependencies | Initialization | Training data | Provisional eligibility | Blocking / action |
|---|---|---|---|---|---|
| C1: SMP U-Net + ResNet34, **random weights** | SMP core MIT; avoid licensed special-case encoders; verify selected version and transitive dependencies | No third-party checkpoint, `encoder_weights=None` as configuration *hypothesis* | `restor/tcd` CC BY subset only, attribution+provenance QA pending | **Candidate** | Freeze package version, check actual module imports and training flow, verify all license notices/data provenance; compute adequacy unknown |
| C2: SMP U-Net + ResNet34, ImageNet pretraining | SMP core MIT; ResNet encoder's effective implementation and deps need confirmation | ImageNet-origin checkpoint: **rights/source and redistribution not yet cleared** | Same | **Conditional / checkpoint unknown** | Audit exact source/model artifact and grant, no assumption library MIT covers weights |
| C3: HF Transformers SegFormer, randomly initialized | File shows Apache-2.0 header; independent provenance and dependencies need specific audit | No externally downloaded model checkpoint, initialize from clean config if viable | Same | **Candidate with IP scope uncertainty** | Audit independence vs NVIDIA-licensed material, downstream files and package versions before selection |
| C4: HF Transformers SegFormer + `nvidia/mit-b0` | As C3 | HF model page labels license **other**, author description credits NVIDIA | Same | **Blocked from eligibility pending explicit rights** | Do not use this NVIDIA checkpoint in a cleared training lineage without permission |
| X1: SMP SegFormer encoder, random or pretrained | **Specific SMP encoder source is noncommercial** | Irrelevant: code restriction already blocks | Same | **Blocked as-is** | Replace with separately cleared implementation if possible |
| X2: Restor published TCD weights Fine-tuned | TCD pipeline Apache overall, dependencies vary | Published model-weight restrictions documented in prior AUDIT_03 | CC BY-only fine-tune does not erase upstream lineage | **Blocked for commercial candidate** | Separate research-only benchmarking if permitted |

The word **Blocked** refers to the *exact identified implementation/checkpoint route*; it does not mean the abstract neural-network architecture is universally unavailable for commercial use. No clean candidate has passed end-to-end license review, and no performance or compute costs have been measured.

## 3. Source cross-checks and caveats

- SMP's official per-file license inventory was **newly identified** in this P0 audit; earlier assumptions that SMP MIT automatically included its SegFormer encoder were too broad.
- `nvidia/mit-b0`: https://huggingface.co/nvidia/mit-b0 currently displays **License: other**; do not interpret it as Apache/MIT permissive weights. Source `https://github.com/NVlabs/SegFormer` uses NVIDIA noncommercial license.
- Torchvision ResNet34 implementation accepts `weights=None` by default in its current source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py . The library's code license is not a grant for every third-party pretrained checkpoint: https://github.com/pytorch/vision/blob/main/LICENSE .
- Hugging Face Transformers repository software uses Apache-2.0: https://github.com/huggingface/transformers/blob/main/LICENSE ; SegFormer model implementation file carries Apache-2.0 header. Separately determine whether a *commercially clear implementation* can be instantiated independently of NVIDIA work, and record legal/compliance assessment if uncertain.
- `restor/tcd` dataset card https://huggingface.co/datasets/restor/tcd identifies CC BY subset, subject to record provenance, attribution and QA per previous audit. Dataset availability does not guarantee correctness of labels or clearance of published weights.
- Avoid extrapolating from upstream component-specific license lists to any uninspected version of a library. Pin package release and dependency lock before G0 PASS.

## 4. G0 acceptance checklist

- [ ] Pin **exact release/commit** of chosen library files, dependencies, model configs, and any weights.
- [ ] Enumerate exact source files imported at runtime; inspect files with different per-component licenses.
- [ ] Record source and license evidence for **each** initialization checkpoint, including encoder weights and Imagenet-related constraints if applicable.
- [ ] Document whether final training weights can be used, distributed, and incorporated into paid analytical services (contract/licensing review as appropriate).
- [ ] Create attribution/notice manifest for software dependencies and OAM-TCD records.
- [ ] Document reproducible zero-restricted-weights startup with no implicit auto-download.
- [ ] Get organizational/compliance review for disputed components (SegFormer independence and third-party checkpoint provenance).
- [ ] Only after above: name at least one route **G0 PASS**, then initiate G1 training-record QA and a minimal experiment.

## 5. Research direction

Prioritize **C1 U-Net/R34 with random initialization** as a *license-traceability investigation*, **not as an adopted training architecture**; it likely has fewer third-party checkpoint issues than pretraining-dependent routes, but may need substantially more optimization/compute. In parallel, investigate **C3 HF SegFormer without NVIDIA pretrained weights** from the standpoint of independent implementation and rights, then evaluate feasibility scientifically after data QA.

**No training, test usage, changes to software, or license clearance claimed.**
