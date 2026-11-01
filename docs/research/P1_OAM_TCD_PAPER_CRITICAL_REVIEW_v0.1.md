# PR #5 — OAM-TCD paper and supplementary material: critical review (v0.1)

**Date:** 2026-11-01  
**Status:** completed *document/literature* review; **not** experimental validation, license acceptance, or model selection.  
**Project question:** What exactly did Veitch-Michaelis et al. (NeurIPS 2024) establish, and what can be safely carried into an OUEM commercial-clean, CC BY-only tree-canopy segmentation model at GSI ~0.5–0.6 m/px?

## 1. Scope and primary sources

1. [Official NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/58efdd77196fa8159062afa0408245da-Paper-Datasets_and_Benchmarks_Track.pdf), especially §§3–6, Table 1, pp. 5–10.
2. [Official supplementary material](https://papers.nips.cc/paper_files/paper/2024/file/58efdd77196fa8159062afa0408245da-Supplemental-Datasets_and_Benchmarks_Track.pdf), dataset card §§1.1–1.3, model card §2.1, Tables 1–4.
3. [Official NeurIPS landing page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/58efdd77196fa8159062afa0408245da-Abstract-Datasets_and_Benchmarks_Track.html).
4. [TCD public pipeline](https://github.com/Restor-Foundation/tcd) to distinguish implementation descriptions from original experimental methods.

**Verification note:** Paper textual extraction and printed p. 7 screenshot (Table 1 plus methods) were checked; supplementary PDF text and tabular sections were inspected, but its oversized PDF could not be visually screenshotted through the browsing service. Accordingly supplementary table numbers below reflect the official text, not independent execution. This review makes no claims that training checkpoints have been individually inspected.

## 2. Dataset construction, target task, and split

| Source-based fact | Implication / caveat |
|---|---|
| 5,072 globally sampled tiles, 2048×2048 RGB at nominal **10 cm/px**, resampled uniformly from OAM surveys | Around **204.8 m square per source tile**, spatial extent depends on GSD; 1024 px random crop spans **102.4 m per side** at 10 cm (about 1.05 ha), **not 100 m²** despite paper wording |
| Binary semantic targets formed by combining *tree* (individual) and *canopy* (groups) instance annotations | Matches OUEM's **non-individualized canopy cover** objective; preserve these semantics and verify crowd/overlap behavior |
| Full set contains CC BY **4,608**, CC BY-NC **272**, CC BY-SA **192** | Difference between all-image and CC BY image counts is 464/5072=**9.15%**; **not** valid to call that training set reduction |
| Official all-license split: **4,406 training, 666 holdout**, 5-fold CV inside training | Source-image (OAM orthomosaic) groups must remain disjoint; do not apply 4,406/666 blindly after removing non-CC BY images |
| Biome-stratified splits; no common source orthomosaic across folds/test | Keep **oam_id** grouping and validate geographic diversity. Biome counts may shift when filtering licenses |
| OAM aerial acquisition seasons not controlled; labels include leafless trees; few negative-only scenes | Urban false positives, deciduous phenology and underrepresented background/negative scenes require targeted Japanese validation |
| Dense canopy groups not subdivided to individual trees | A shortcoming for instance counting, **not** automatically a problem for unioned binary semantic cover |
| Most OAM-TCD labels produced by paid non-ecologist annotators and reviewed; biases/inconsistencies acknowledged | Source-label quality is imperfect. Field and GSI-image QA necessary, especially ambiguous shrubs/trees |

**Important contextual detail:** the authors explicitly say that 2048px tiles could be downscaled to **256px / 0.8m** for research on near-meter imagery (paper §3.2); this describes **possible** utility, not evaluated near-meter performance.

## 3. Models, initialization, and hyperparameters

The original paper §4 states **all** benchmark model families were **fine-tuned from ImageNet- or MS-COCO-pretrained models**, and reports quicker convergence than scratch starts. This is not a factorial or controlled experiment reporting the eventual final-IoU gap between pretrained and scratch training; **the exact initial checkpoint per family is not identified by that sentence**.

| Family | Implementation / initial source | Task | Loss and notes |
|---|---|---|---|
| **U-Net ResNet34/50** | Segmentation Models PyTorch, ImageNet/MS-COCO initialization collectively reported | Binary semantic | Focal loss |
| **SegFormer MiT-B0…B5** | SegFormer, ImageNet/MS-COCO initialization collectively reported | Binary semantic | Cross-entropy; paper cautions on NVIDIA licensing |
| **Mask R-CNN R50** | Detectron2, ImageNet/MS-COCO initialization collectively reported | Individual/group instance | Mask R-CNN objectives; this family's AP metrics cannot be compared to semantic IoU |

Shared semantic training conditions (paper §4 and supplement Table 2): nominal **10 cm/px**; random **1024×1024** crops from 2048²; ImageNet mean/std normalization; flips, rotations, blur and color adjustments (the detailed supplemental list omits blur—paper includes it); AdamW; learning rate **1e-4 to 1e-5**; reduce-on-plateau schedule; batch **32**; **75 epochs** for cross-validation and **50 epochs** for final models; seed **42**. **Caveat:** these are *fine-tuning* settings on RTX3090, **not scratch-training defaults**. Applying a 1024px crop never changes GSD.

Instance training conditions (paper §4 and supplement Table 3): Mask R-CNN, Detectron2, 1024², AdamW learning rate **0.001**, batch **8**, **100,000 iterations**, step 0.1 at 80k/90k, maximum predictions increased to 512.

## 4. Experimental results on native OAM-TCD

### Semantic segmentation: original paper Table 1

| Model | 5-fold mean IoU ± SD | Holdout IoU | Holdout F1 |
|---|---:|---:|---:|
| U-Net ResNet34 | 0.842 ± 0.004 | **0.838** | 0.871 |
| U-Net ResNet50 | 0.856 ± 0.007 | **0.849** | 0.880 |
| SegFormer MiT-B0 | 0.858 ± 0.011 | **0.865** | 0.882 |
| SegFormer MiT-B1 | 0.873 ± 0.009 | **0.870** | 0.891 |
| SegFormer MiT-B2 | 0.878 ± 0.013 | **0.871** | 0.898 |
| SegFormer MiT-B3 | 0.879 ± 0.009 | **0.875** | 0.875 |
| SegFormer MiT-B4 | 0.880 ± 0.007 | **0.875** | 0.901 |
| SegFormer MiT-B5 | 0.887 ± 0.003 | **0.876** | 0.902 |

**Interpretation:**
- Holdout IoU B0−U-Net R34 = **+0.027**; B5−B0 = **+0.011**; B5−U-Net R34 = **+0.038**. Do not portray small differences as a universally controlled architecture advantage; family-specific pretrained weights/losses and parameter budgets differ.
- Larger SegFormer models show diminishing holdout IoU gains. This suggests *evaluating a modest network before paying the cost of a larger one*, not assuming U-Net will meet OUEM accuracy targets.
- The paper hypothesizes that annotation consistency/diversity is limiting plateau; this is an **author hypothesis**, not an ablation establishing the cause.
- These native 10cm reference-model scores do **not** predict CC BY-only scratch results or Japanese GSI performance.

### Instance segmentation

Mask R-CNN R50 **mAP50 segm**: CV **41.79 ± 1.38**; holdout **43.22** (paper §4). These are **not** semantic canopy IoU or tree-count accuracy and are outside OUEM's immediate objective. The authors found transformer-based alternatives harder to train reliably / more memory intensive in their experiments (not a universal result).

## 5. Independent geographic validation: especially relevant for OUEM

**Zurich (Switzerland)**, paper §5: 2022 Swisstopo **10cm/px** RGB imagery over city (~90 km²), compared with a LiDAR-derived canopy height map and municipal inventory. For **SegFormer MiT-B5 versus CHM-derived canopy**: **IoU 0.791; accuracy 0.922; F1 0.883**. The CHM may have classification or resolution limitations and date mismatches may apply; this is **not** perfect tree ground truth or a Japanese near-meter validation. Keypoint recall for Mask R-CNN **0.36 individual-tree only / 0.73 when group canopy accepted** (inventory non-exhaustive for private trees, so false positive rate not calculable). The strong difference highlights why binary canopy cover—not exhaustive individual tree counting—is the right first output for OUEM.

**Tonga WeRobotics**, paper §5: 325 ha mosaic, 13,402 keypoints of four species; Mask R-CNN recall **0.64 individual-tree / 0.94 when group canopy accepted**. The annotation is incomplete and task is instance-level; do not report these as semantic IoU or as proof of low-resolution generalization.

**Useful methodological analogy:** combining RGB-only canopy segmentation with an independent CHM is supported as a practical *comparison/filtering* workflow in the paper. It does **not** imply CHM directly fixes every RGB classification error or that CHM is a perfect GT.

## 6. Resolution, domain shift, and error modes

- Paper and supplemental model card expressly caution that models trained at **10cm/px** may be unreliable at substantially different GSD. **Fine-tuning on target resolutions is recommended** (supplement §2.1, Out-of-Scope Use).
- The 0.8m downscale suggestion (§3.2) is a **suggestion**, without direct benchmark evidence of a corresponding model's accuracy.
- Principal false positives noted in supplement: **large bushes, shrubs or ground vegetation resembling tree crowns**, particularly away from trained GSD. This directly informs GSI error analysis.
- No target GSI 0.5–0.6m adaptation experiment, Japan independent benchmark, scratch-vs-pretrained final accuracy ablation, CC BY-only vs mixed-license accuracy ablation, or canopy-boundary assessment at GSI scale is reported.
- Geometric effects: 2048→256 by 8× shrink yields near-meter GSD, but OUEM experiment must define target GSD, RGB area interpolation and GT area-fraction threshold *together*, and not confuse resampling with new information.

## 7. Compute: avoid misuse of the runtime table

- Original reported hardware: **single NVIDIA RTX3090 24 GB** + AMD Ryzen 7950X + 64 GB RAM, supplement §2.1.
- Supplement Table 4: 6-run **aggregate estimates** (5-fold CV + one final): UNet R34 **33h**, SegFormer B0 **33h**, UNet R50 **90h**, SegFormer B5 **150h**, Mask R-CNN **126h**. Table is for power/emissions estimation, not a strict reproducibility benchmark; does not include failed experiments and can differ from wall-clock measurements.
- No defensible numerical extrapolation to **RTX A2000 12GB** or scratch from these sources alone. Benchmark A2000 with actual patch shape, AMP, batch size, peak VRAM, tiles/sec, and resume once the model/data gate is met.
- Checkpoints, deterministic group splits, training logs and stopping criteria matter more than choosing an unsupported number of training hours.

## 8. Commercial-use delineation

- **Dataset provenance:** use only independently verified **CC BY 4.0** source imagery/labels, record attribution and group identity. Main paper states CC BY-only imagery should generally perform well; this is an author assessment, **not an empirical matched-condition accuracy ablation**.
- **Code:** OAM-TCD pipeline overall Apache-2.0, but that does not retroactively clear each dependency or vendor implementation.
- **Published final weights:** supplement model card states **U-Net & Mask R-CNN CC BY-NC 4.0**, SegFormer under **NVIDIA SegFormer Source Code License**; do not reuse for commercial deployment without rights resolution.
- **Pretraining checkpoints:** original exact starting weight IDs and their rights are **not established** by the paper; keep as *unresolved separate audit*. Do not assume all ImageNet weights are prohibited or allowed.
- **Working project baseline remains** CC BY-only training **from random initialization** until a separate approved checkpoint provenance review changes it.

## 9. Decision-impact matrix for OUEM

| Research question | Supported conclusion | Decision for current project |
|---|---|---|
| Is semantic cover appropriate when trees touch? | Yes; paper merges individual and group into binary tree cover | **Retain binary semantic target**, no individual-tree requirement |
| Is 4,608 CC BY-only subset too small? | Authors suggest CC BY-only can work, but **no controlled accuracy result** | **Retain CC BY-only**; audit per-license spatial/biome balance and grouped splits |
| Does 9.15% fewer *total* tiles imply similar accuracy? | No direct evidence, though plausible as hypothesis | Do not claim equivalence; validate directly |
| Are the pretrained authors' model results a scratch benchmark? | No; every model family fine-tuned pretrained weights | Do not set native benchmark IoU as acceptance target for scratch |
| Is SegFormer B5 worth the training/licensing burden? | Only +0.011 holdout IoU vs B0, under reference conditions | Prefer testing smaller cleared models; **no model selected yet** |
| Do authors validate 0.6m GSD? | No; explicitly warn about changing resolution | **GSD adaptation & Japanese evaluation remain central** |
| Can CHM help check masks? | Zurich comparison and false-positive discussions suggest yes | Include CHM-based diagnostics, with independent accuracy caveat |
| Is A2000 adequate? | Not evaluated by authors | Benchmark actual workload; no runtime promise |
| Have we cleared commercial release? | No | **G0 remains NOT PASSED** |

## 10. Next-stage scope after PR #5 review

1. P0/G0: choose viable **permissively implemented architecture**; audit exact source tag and all packaging/dependency licenses. Evaluate pretrained weights only as a *separate* opt-in licensing investigation.
2. P1/G1: CC BY-only true pixel/mask QA, source-level license attribution, license-filtered biome histogram, spatial/oam_id grouped train/val and final held-out data, reproducible transforms.
3. P2/G2: matched CC BY-only scratch model at native 10cm and artificially degraded 50–60cm; test 0.5–0.6m Japanese GSI separately. Track Tree IoU/F1/precision/recall, canopy-cover bias, small crowns and boundaries; do not reuse GT54 until independent final evaluation as previously agreed.
4. P3/G3: georeferenced outputs, tiling overlap, nodata, and canopy–CHM integration into OUEM.

### Disposition

**Literature review complete within cited scope.** This PR can be accepted **as research documentation only**; no executable changes, no real-data validation, no architecture selection, and no license gate pass. Individual initial-checkpoint license tracking is explicitly deferred to a future task.
