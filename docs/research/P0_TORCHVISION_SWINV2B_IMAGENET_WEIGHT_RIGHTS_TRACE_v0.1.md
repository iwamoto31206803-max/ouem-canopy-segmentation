# Torchvision Swin-v2 Base ImageNet-1K checkpoint: primary-source commercial-use review v0.1

Reviewed: 2026-11-01. **User scope**: no external publication or redistribution of learned weights or trained OUEM model; potential internal fine-tuning and commercial inference. User accepts Ai2's expressly published Satlas ODC-BY designation as working presumption while documenting upstream uncertainty. **No downloads, execution, or external inquiries. Not legal advice.**

## Checkpoint identity — pinned to code

- The Satlas aerial configurations [`configs/aerial/swinb_si.txt`](https://github.com/allenai/satlas/blob/main/configs/aerial/swinb_si.txt) and [`configs/aerial/swinb_si_pretrain.txt`](https://github.com/allenai/satlas/blob/main/configs/aerial/swinb_si_pretrain.txt) specify `Backbone: swin`, `Arch: swin_v2_b`, `Pretrained: true`.
- The Satlas constructor [`satlas/model/model.py`](https://github.com/allenai/satlas/blob/main/satlas/model/model.py) calls `torchvision.models.swin_v2_b(weights=torchvision.models.swin_transformer.Swin_V2_B_Weights.IMAGENET1K_V1)` for that path, then **replaces the input Conv2d**. Thus ImageNet-derived initialization is evidenced for the published *code/config path* but unchanged retention of every parameter is not.
- PyTorch's official [Swin-v2 Base model documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_v2_b.html) identifies **`Swin_V2_B_Weights.IMAGENET1K_V1`**, ImageNet-1K top-1 84.112%, 87,930,848 parameters, ~336.4 MB. Official [torchvision source](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py), blob SHA `80850b4a389e6488a2d5ae76a4159b5ad26a6faa`, defines URL **`https://download.pytorch.org/models/swin_v2_b-781e5279.pth`** and model recipe `https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2`. Filename suffix `781e5279` is a short hash-like identifier, **NOT a verified full-file SHA256**. No actual binary downloaded or hash computed.
- The specific `WeightsEnum` entry provides transform/accuracy/recipe/URL/size metadata but **no specific pretrained-weight license declaration** in that entry. **Absence of a declaration is not an affirmative commercial-use license nor an explicit restriction.**
- Torchvision [source package license](https://github.com/pytorch/vision/blob/main/LICENSE), file SHA `1edcf92c3317b90fedd187e2eaad101bd1c1efc5`, is **BSD-3-Clause**. **Do not extrapolate code license to particular pretrained weight**.
- PyTorch's [Models and pretrained weights docs](https://docs.pytorch.org/vision/master/models.html), note in "General information on pre-trained weights," states models **may** have their own licenses/conditions derived from datasets, and that checking rights for specific use is **the user's responsibility**. This is a **general caution**, not a declaration that Swin-v2 Base weights are NC.
  
## ImageNet original terms: specific clauses

The first-party [ImageNet Access Agreement](https://image-net.org/accessagreement) is available without authentication. Relevant clauses, paraphrased:

1. Clause 1 requires the **Researcher to use the Database solely for non-commercial research and educational purposes**.
2. Clause 2 states the institutions provide **no warranties**, including non-infringement.
3. Clause 3 makes the Researcher responsible for use of the **Database** and copies of copyrighted images.
4. Clause 4 requires colleagues provided database access to accept the same terms.
5. Clause 5 reserves termination of database access.
6. **Clause 6:** if the Researcher works for a **for-profit commercial entity**, that employer is **also bound by the agreement**; the Researcher represents authorization to bind that employer.
7. Clause 7 selects New Jersey law for disputes.

The official [ImageNet About page](https://www.image-net.org/about.php) states ImageNet itself **does not own copyright over component images**. The [download page](https://www.image-net.org/download) displays same access terms.

**Critical distinction:** these are conditions for actors requesting ImageNet **database access**. In the reviewed official access agreement, terms do **not expressly address contractual inheritance by unrelated third-party downstream users of `torchvision` weights or commercial inference on a Satlas-published artifact**. It is equally incorrect to infer **automatic NC on all downstream weights** or **automatic unconditional permission**. Whether contractual or copyright/database rights extend further is a question for legal examination; no binding legal conclusion is made here.

**Operational rule:** OUEM should **not directly download or train on ImageNet dataset under that standard agreement** for commercial development. The user is instead considering weights already published by Ai2, without obtaining the ImageNet dataset.

## Publisher's independent grant

Ai2 explicitly states in [SatlasPretrain overview](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md) and [satlaspretrain_models README](https://github.com/allenai/satlaspretrain_models/blob/main/README.md) that pretrained *model checkpoints/weights* are published under **ODC-BY**. Ai2's [DataLicense](https://github.com/allenai/satlas/blob/main/DataLicense) includes commercial-use permission for covered database rights, subject to credit; its precise application to ML weight artifacts is an interpretive caveat. The aerial RGB single-image ID is **`Aerial_SwinB_SI`**, 0.5–2m/px. The official loader also exposes backbone+FPN and randomly initialized segmentation head. For internal commercial inference the publisher declaration is the most directly relevant positive published permission, though there is no explicit weight-specific third-party non-infringement warranty.

## Evidence-based evaluation

| Proposition | Result |
|---|---|
| Exact torchvision upstream weight name/URL identified in code path? | **YES**, model constructor plus exact source URL |
| Public Torchvision entry specifically labels that weight NC? | **NOT FOUND** in reviewed source/entry; not proof of permissiveness |
| ImageNet database standard agreement permits commercial training on its images? | **NO**: clause 1 prohibits; clause 6 expressly extends to employer of commercial entity's researcher |
| ImageNet agreement expressly forbids all *downstream* commercial inference using separately distributed weights? | **NOT STATED** in reviewed clauses; no claim of legal permission implied |
| Ai2 explicitly labels Satlas published final weights ODC-BY? | **YES** |
| User needs redistribution rights for OUEM checkpoint? | **NO** under declared plan; internal use / attribution still relevant |
| Exact released Ai2 binary's historical training run and full hashes independently verified? | **NO** |
| Definitive legal opinion on commercial deployment? | **NO** |

## Remaining narrow gaps and follow-on work

1. Check PyTorch's [Swin-v2 reference training recipe](https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2) for train-data and reproducibility detail; avoid conflating pretraining recipe with license terms.
2. Seek any **weight-specific** official license or clarification on `swin_v2_b-781e5279.pth`; generic torchvision README/library license insufficient. If no such statement available, record **unknown**.
3. Optionally request from Ai2 (only with user authorization): confirm **commercial internal fine-tuning and inference with no checkpoint redistribution**, despite ImageNet initialization; any notice/attribution requirements.
4. If/when actually downloading artifacts, capture immutable URL, checkpoint version, **full-file SHA256**, notice text, package version, dated copies of official license pages. No download performed in this documentation pass.

**Working recommendation:** continue to treat Satlas as a promising experimental candidate based on explicit Ai2 ODC-BY. **Do not** characterize all ImageNet-derived downstream models as automatically prohibited or as automatically commercially cleared. Keep production rights acceptance and performance/VRAM gates separate.
