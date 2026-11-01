# ImageNet -> torchvision -> SatlasPretrain: rights and use-scope audit (v0.1)

Date: 2026-11-01. Scope confirmed by user: **OUEM will NOT publish or redistribute trained model weights or models.** Intended possible future use is **commercial/internal inference**, with fine-tuning during development. **Accept Ai2's express ODC-BY release as the working presumption**, but investigate upstream ImageNet rather than claim complete certainty. No external inquiry or downloads made. Not legal advice.

## Primary-source findings

1. **ImageNet terms are dataset access terms.** [ImageNet official download](https://www.image-net.org/download) says a researcher requesting access to the **Database** agrees to use the Database only for **non-commercial research and educational purposes**. [ImageNet official about page](https://www.image-net.org/about.php) says ImageNet **does not own copyright in the constituent images**. This is a clear limitation on access/use of the ImageNet *database*, but the quoted text **does not explicitly state that all model weight files subsequently trained on it are forever non-commercial**. Do not infer that conclusion from the dataset text alone. A separate contractual issue may arise for the person/entity who downloaded and trained with the data; that is not necessarily identical to the downstream user's obligations.
2. **Torchvision explicitly does NOT clear every pretrained weight.** [Official PyTorch Torchvision models docs](https://docs.pytorch.org/vision/master/models.html) state pretrained models *may* have their own licenses or conditions derived from the training dataset, and that users are responsible for checking intended use. [Official torchvision README](https://github.com/pytorch/vision/blob/main/README.md) names **SWAG weights** as explicitly **CC BY-NC 4.0**; this shows source-code license cannot be blanket-applied to all weights. The official wording does **not itself assign CC BY-NC to `Swin_V2_B_Weights.IMAGENET1K_V1`**.
3. **Satlas aerial uses ImageNet initial weights in published source/config.** [Ai2 aerial config](https://github.com/allenai/satlas/blob/main/configs/aerial/swinb_si.txt), blob `9e21a7db2a44248530b3f80aac10d822e231beb5`, says `swin_v2_b`, `Pretrained:true`. [Ai2 training-specific config](https://github.com/allenai/satlas/blob/main/configs/aerial/swinb_si_pretrain.txt), blob `b2e38da43d7bb1a1a2baa4af192fecaeab2a2783`, same. [Ai2 `satlas/model/model.py`](https://github.com/allenai/satlas/blob/main/satlas/model/model.py), blob `d3200b47f23a837316116b9c0f83255a3b88b34d`, explicitly instantiates `torchvision.models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)` when `Pretrained:true`. It then replaces first Conv2d with a new layer, so original weights are only partly retained. These are source/config observations; historical proof for exact published binary's training run remains outstanding.
4. **Ai2's final checkpoint license is explicit.** [Ai2 SatlasPretrain overview](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md) and [Ai2 checkpoint library](https://github.com/allenai/satlaspretrain_models/blob/main/README.md) both state the **model weights/checkpoints** are published under **ODC-BY**; [license text](https://github.com/allenai/satlas/blob/main/DataLicense) permits commercial database use with attribution, to extent licensed rights cover it. ODC-BY is a database license, so legal interpretation for model-weight artifact rights and upstream material still merits caution. This does not amount to a legal warranty from Ai2.

## What the evidence does *not* prove

- **Not proven:** ImageNet's non-commercial *dataset* access clause automatically applies to all third-party **downstream pretrained weights**, fine-tuned weights, commercial inference services, or inference outputs.
- **Not proven:** Ai2's ODC-BY designation necessarily extinguishes all independent upstream claims, if any.
- **Not proven:** Satlas's exact distributed `Aerial_SwinB_SI` binary can be reconstructed from the currently published config; exact historical checkpoint ID, revision, hash and original training-run documentation need a reproducibility audit.
- **Not proven:** Using Satlas in a commercial inference workflow is legally prohibited. The first-party public license is positive evidence for a permissible path.
- **Not needed for user's planned scope:** broad redistribution or publication rights analysis for downstream OUEM weights. Do not use redistributability as the deciding blocker. However attribution and possible internal-use conditions remain relevant.

## Practical legal risk matrix for user's intended scope

| Question | Evidence level | Practical treatment |
|---|---|---|
| May the user download/use ImageNet **dataset** for commercial training? | Official terms restrict access to noncommercial research/education | **Do not download or train on ImageNet dataset** in OUEM |
| Does `torchvision` BSD-3-Clause source license clear ImageNet pretrained **weights**? | No; torchvision itself explicitly cautions | Do not claim source license clears weight |
| Is Satlas final weight officially ODC-BY? | Yes, direct Ai2 docs | **Presume grant in good faith** and archive official notice |
| Does the dataset clause explicitly forbid commercial use of a downstream Satlas model? | Not in reviewed dataset access text | **No affirmative prohibition established**, but not proof of unrestricted rights |
| Does user's **no-redistribution** plan eliminate all possible concerns? | No; different rights/terms can govern internal commercial use | Reduces distribution-specific obligations, not all use risk |
| Is production/legal clearance complete? | No | Keep **commercial rights review open**, distinct from technical evaluation |

## Proportionate next steps

**No need to require impossible complete proof of every training image.** Instead:
1. Record exact Ai2 ODC-BY declaration and downloaded model identifier/version at time of eventual use; retain NOTICE/attribution. **No need to download now.**
2. Verify official **torchvision Swin_V2_B_Weights.IMAGENET1K_V1** metadata and the exact weight artifact, including whether any explicit separate use restriction exists. The Torchvision warning is general, not a checkpoint-specific denial.
3. Read any first-party ImageNet FAQ/legal statements about **trained weights** as opposed to dataset access. If none, mark **not addressed by public text**; do not invent an affirmative permission or restriction.
4. For a higher-assurance commercial gate, **optional** bounded written inquiry to Ai2: “For your ODC-BY Aerial_SwinB_SI weights, may an organization fine-tune internally and use commercially for inference without redistributing model weights, despite ImageNet-initialized torchvision Swin-v2-B in published training configs? Any extra terms or attribution?” Ask only if user authorizes contacting publisher.
5. Keep experimental (technical) gate independent: GSI quality, performance vs scratch, and hardware feasibility. User's intended internal use should be documented; any external inference output/API access deserves later separate assessment if scope changes.

**Decision:** **Proceed with Satlas as an eligible *research/benchmark candidate* under publisher's published ODC-BY terms; no blanket commercial production guarantee and no finding of a categorical ImageNet-derived prohibition.** This is a risk-based scope-specific judgment, not a legal opinion. G0 remains NOT PASSED for technical and commercial-production release.
