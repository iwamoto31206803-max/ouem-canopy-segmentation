# M1-2B rights, semantics and spatial audit

## Evidence and rights recommendation

Audit date 2026-11-05. Dataset and reference code revisions remain those in the [sample manifest](M1_2A_SAMPLE_MANIFEST.json). M1-2A raw hashes were revalidated before diagnostics.

The [paper section 3.4](https://arxiv.org/html/2407.11743v1#S3.SS4) explains redistribution under the source metadata license and separation by license. The [fixed dataset card](https://huggingface.co/datasets/restor/tcd/blob/d97d4da0ebbb6e249ae95ac5e19656babd972eb2/README.md) declares CC BY 4.0 and both row licenses are CC-BY 4.0. The [official dataset documentation](https://github.com/Restor-Foundation/tcd/blob/05dab754a3fd15b83d6b7606de7310155f4e2df2/docs/dataset.md#license) declares main annotations CC BY 4.0 and explains imagery source licenses. These are affirmative publisher declarations supporting internal data preparation and training.

The legacy OAM S3 metadata URLs failed with HTTP 403 during M1-2A. A [documented official STAC route](https://docs.imagery.hotosm.org/dev/ingest/backfill/) resolved both OAM IDs successfully during M1-2B.

- Image 661 resolves through https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/60f0a29989ec130007b828a8
- Image 1972 resolves through https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/5bba1c4f9ed15b0006d24f3b

Both exact item IDs match the dataset oam_id. Each declares CC-BY-4.0 and provides a named producer/licensor. Each visual asset retains the same source directory and source image identifier as the corresponding legacy metadata URL, on the oin-hotosm-temp bucket. This resolves the source identity and attribution retrieval gap. Provider names and contacts remain in externally cached STAC records under the user's restriction on committing personal information. Hashes and canonical metadata URLs are recorded in the public manifest. Retain that external attribution record when using either sample and prepare appropriate credits at handoff.

Proposed evidence categories for review are VERIFIED (consistent source and dataset license, attribution recovered), SOURCE_DECLARED_PENDING_ATTRIBUTION (publisher/row declaration established but original attribution unavailable), and HOLD (contradiction or insufficient evidence). Both samples now meet the proposed VERIFIED evidence category. This proposal concerns dataset use within the documented project scope. Organizational commercial production rights review remains a separate later gate.

## Label semantics recommendation

The [official converter](https://github.com/Restor-Foundation/tcd/blob/05dab754a3fd15b83d6b7606de7310155f4e2df2/tools/generate_dataset.py) hardcodes Canopy 1 / Tree 2 and binarizes annotation by nonzero pixel content without filtering iscrowd. Both samples' category-validated masks are exactly equal to that official binary convention.

Recommend ID 0 as supervised background under this dataset convention and IDs mapped to category 1 or 2 as positive, including category-known crowd pixels. Unknown nonzero IDs/categories remain errors. ID 0 represents absence of annotated tree coverage. Label omissions and ambiguous vegetation boundaries can remain. Ecological completeness is unverified. The paper describes conservative annotations and the dataset documentation acknowledges incomplete annotation of other land-cover classes.

COCO crowd flags describe grouped/non-individual instance representation, with RLE storage supported by the [COCO mask implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py). The flag retains a category and the official binary export includes its pixels. In 1972, segment 106343 has category Tree 2 and iscrowd 1. Independent decoding of its uncompressed COCO RLE in Fortran order produced 37,276 pixels and zero differences against panoptic ID 106343. The segment's reported area field is zero despite nonempty raster coverage, so area metadata must not be used to discard this segment.

Both RGB originals are RGB TIFF and annotations lossless PNG. Neither TIFF exposes GDAL_NODATA tag 42113, nor a conflicting orientation tag. No explicit nodata mask was identified in these embedded three-channel records. A separate source validity mask has not been recovered. For these two visually valid tiles, recommend the official whole-tile background convention. Future tiles with padding/nodata need explicit validity handling before CE loss. M1-1 validated clean 0/1 targets only. Maintain long targets [B,H,W], classes 0/1, exact RGB grid matching, and finite loss. Void/ignore policy for future invalid tiles requires its own verification.

## Spatial and pixel observations

The local-only overlay script rehashes originals, regenerates category-based masks, compares saved masks and produces a side-by-side RGB and 35% magenta overlay. Both 2048 x 2048 grids match. EXIF orientation is absent for RGB and annotation in both cases. No geometric transform is applied to align them.

Both overlays were actually viewed at a 1600 x 800 overview. Image 661 shows canopy regions following the woodland and isolated crowns, with the main building, road and open grass mostly excluded. Image 1972 shows forest and sparse crowns aligned over a leaf-off/mixed vegetation scene. No obvious gross shift, rotation, reflection or raster corruption was visible. Individual crown boundary accuracy and small omissions remain unquantified. This is a visual gross-alignment check; no subpixel registration estimate was made.

The original annotation was decoded at full resolution. Uniform-mask handling remains covered by M1-2A tests. Full polygon rasterization was unnecessary for this bounded check. The crowd RLE received the independent check above.

## Sample decisions and handoff

| Sample | Pixel QA | Spatial QA | Label semantics | License evidence | Attribution | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| 661 | verified | visual gross alignment verified | official binary convention supported | dataset + row + exact STAC agree | recovered externally | eligible for assigned train split on review |
| 1972 | verified | visual gross alignment verified | crowd positive supported, RLE exact | dataset + row + exact STAC agree | recovered externally | eligible for assigned val split on review |

The manifest preserves training_eligible=false until this policy recommendation is reviewed. It records resolved evidence separately, replacing the former blanket pending classification. No additional evidence blocker was found for these two samples under the recommended convention. Wider data acquisition still requires per-record rights, validity and group checks. GSI GSD and resolution transforms follow their separate audit.

## Reproduce and external artifacts

```powershell
.\.venv\Scripts\python.exe scripts/m1_2b_overlay_qa.py --cache C:\dev\ouem-data-cache\m1-2a-audit-02
.\.venv\Scripts\python.exe -m pytest -q
```

Existing dependencies suffice. The diagnostic script writes only m1-2b-overlay.png beside each cached pair. Raw STAC JSON remains in the external cache root. No image or raw metadata is included in Git. The manifest records STAC snapshot hashes for audit reproducibility, since the live API can change.

Final offline pytest completed with 19 passed in 5.57 seconds. The local diagnostics were rerun successfully, including exact crowd RLE comparison.
