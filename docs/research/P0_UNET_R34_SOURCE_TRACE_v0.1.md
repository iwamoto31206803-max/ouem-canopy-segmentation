# P0 U-Net ResNet34 implementation and initialization trace — source audit v0.1

**2026-10-31; research only.** Companion to `P0_COMMERCIAL_ROUTE_MATRIX_v0.1.md`. No runtime tests, no legal clearance and no model chosen. GitHub sources inspected at repository default branches; file blob hashes below are observations, not immutable repo commit IDs.

## 1. Exact candidate construction (design hypothesis, not run)

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,       # MUST override the default "imagenet"
    in_channels=3,
    classes=1,
    activation=None,            # output logits; sigmoid applied in evaluation as appropriate
)
```

Source: [SMP U-Net model](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/unet/model.py), blob `5b2212658060996fcf186b5bda99f85164e33636`. Its **constructor default is `encoder_weights="imagenet"`**, which is unsuitable as an implicit default in the candidate. The `weights=None` argument flows to [`get_encoder`](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/encoders/__init__.py), blob `fa013ed75d0e61865374b14aec717ce99ab82cb3`.

For non-`tu-` encoders, `get_encoder` only calls `hf_hub_download` and loads checkpoint state when `weights is not None`. Thus **source inspection indicates zero pretrained-weight download for this particular `resnet34` configuration**, not measured offline behavior. The encoder subclass [`ResNetEncoder`](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/encoders/resnet.py), blob `d4f2db4e0d581629015e36789d6e18829bca2412`, subclasses `torchvision.models.resnet.ResNet` and imports `BasicBlock`. Code-based inference: normal PyTorch random initialization, subject to actual environment test.

## 2. New license-compliance complication: package eager imports

The same `segmentation_models_pytorch/encoders/__init__.py` source executes:
```python
from .mix_transformer import mix_transformer_encoders
from .mobileone import mobileone_encoders
```
even when the user selects `resnet34`.

The official [component-specific license inventory](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/licenses/LICENSES.md), blob `e51ad8d0a6755b6c403bef636ddff9e355428a19`, explicitly states the `mix_transformer.py` SegFormer encoder is **noncommercial**. The package-wide [MIT LICENSE](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE) cannot be treated as the license of that file. **It is not yet established whether merely importing, packaging, or distributing unused noncommercial code has any bearing on a particular planned application.** Do not state that the random ResNet route is commercially cleared, and do not overstate this as an established legal violation. Distinguish (1) running only ResNet weights/ops, (2) loading other modules at import, (3) copied bundled/distributed files, and (4) contract or copyright consequences. This needs version-specific legal/compliance review.

Potential alternatives needing evaluation:
- Use a clearly scoped, permissively licensed U-Net implementation and torchvision ResNet encoder, without copying/carrying the SMP `mix_transformer.py` module. Do not assume a homemade copy of SMP automatically removes its license obligations; audit actual copied code and dependencies.
- Investigate whether a known permissible SMP release provides a conditional/lazy-import boundary, but **no such release was verified** here.
- Obtain permission/clarification if retaining distribution of SMP and its licensed components.

## 3. Pretrained checkpoints are independent risk

ResNet34 `imagenet` is mapped by the inspected SMP code to HF repo `smp-hub/resnet34.imagenet`, revision `7a57b34f723329ff020b3f8bc41771163c519d0c`. That checkpoint is NOT used in the `None` construction path. Its rights **have not been verified**, so pretrained C2 remains ineligible for G0 PASS.

## 4. Dependencies and version pinning

The [SMP pyproject](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/pyproject.toml), blob `93961682af146d0a4809286c62d1e3f1a0a201a7`, requires Python >=3.11 and lists `torch>=2`, `torchvision>=0.15.1`, `timm>=0.9`, `huggingface-hub>=0.24`, `numpy`, `pillow`, `safetensors`, `tqdm`; direct version ranges are NOT a reproducible lock. Both import dependency and transitive package licenses still require inspection. Target local Python 3.11 is broadly compatible with declared Python minimum; no installation performed.

## 5. G0 candidate status

| Criterion | Source-level outcome | G0 |
|---|---|---|
| Explicit ResNet34 U-Net model constructor | Found in current SMP implementation | Partial |
| `encoder_weights=None` bypasses pretrained download branch | Source-confirmed for `resnet34` path | Partial; runtime offline test outstanding |
| Entire selected source/dependency license path clearly commercially usable | **Not verified: eager import of noncommercial mix_transformer** | BLOCKED/UNKNOWN |
| ImageNet checkpoints omitted from clean lineage | Planned config, not actual run | Pending |
| Training imagery/annotation CC BY provenance | Previous audit only, real data pending | Pending |
| Practical train speed/memory and accuracy | Not measured | Pending |
| Final checkpoint/commercial distribution permissions | Not reviewed | Pending |

**Result:** The source-level random-initialization mechanism is plausible; **G0 remains NOT PASSED**. The next efficient research step is to identify a minimally scoped permissive U-Net+ResNet34 implementation path without SMP's noncommercial SegFormer file, plus specific dependency rights audit, and compare with a direct PyTorch+torchvision U-Net encoder implementation. Do not begin training or merge the draft until reviewed.
