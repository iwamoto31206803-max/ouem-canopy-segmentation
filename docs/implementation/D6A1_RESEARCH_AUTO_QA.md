# D6-A1 research auto-QA

`scripts/d6a_research_qa.py` is an offline decision and crop-recipe tool for the published OAM-TCD Zenodo release. It reuses the existing documentary rights and geographic checks, upstream binary label decoder, 0.50 m grid and area reducer, and source-seeded 224-pixel selector. It writes one new JSON report. TIFF/PNG inputs are read-only, and the temporary crop arrays stay in memory.

## Input contract

Prepare a JSON contract and pin its SHA from the approved policy inputs. The paths are resolved relative to a local Zenodo cache and a separate local STAC evidence directory. Replace pins and example source IDs before use. The example omits optional candidate `source_hashes`, which, when available, must contain both `rgb` and `annotation` SHA-256 values. The report marks unpinned source hashes as locally observed.

```json
{
  "schema": "d6a-research-policy-v1",
  "source_generation": "zenodo-11617167-v1.0.0",
  "seed": 0,
  "catalog_sha256": {
    "train": "<archives/train_meta.json SHA-256>",
    "test": "<archives/test_meta.json SHA-256>",
    "coco": "<archives/dataset/holdout/train.json SHA-256>"
  },
  "candidates": [
    {
      "image_id": 123,
      "oam_id": "example-source",
      "validation_fold": 1,
      "stac_sha256": "<example-source-stac.json SHA-256>"
    }
  ]
}
```

The catalog must be the approved official metadata snapshot. The contract designates the only candidates inspected. Official Train fold 0 stays Val, folds 1–4 stay Train, and official Test is used only as metadata for split checks. No GT54 inputs are used. The separation check covers this catalog's tile envelopes and metadata object aliases, including the existing 1 km proximity guard. Duplicate pool groups are excluded as a set, independent of order. Full scene extents, unlisted sources and legal clearance remain unverified.

The report distinguishes `MATCHED_EXTERNAL_EXPECTED_SHA256`, `EXTERNAL_EXPECTED_SHA256_MISMATCH`, `SOURCE_AUTHENTICITY_NOT_EXTERNALLY_VERIFIED` (locally observed hashes only), and `NOT_CHECKED` (source bytes never reached). A match checks the supplied contract pin. For A2, use SHA-256 pins derived mechanically from the exact archive members after verifying the archive against the publisher receipt. Preserve the archive/member receipt separately. If unavailable, leave the pins absent and retain the unverified status.

```powershell
python -B scripts/d6a_research_qa.py --root <cache> --evidence <local-stac-directory> `
  --contract <approved-policy.json> --contract-sha256 <reviewed-sha256> --output <new-private-report.json>
```

Use a fresh output outside the repository, cache and evidence directories. Exit 1 indicates a technical failure or invalid preflight. A written report with exit 0 can include HOLD or EXCLUDE.

## Decision rules

- `source_decision` checks designated identity, fold, rights snapshots, known geographic separation, file hashes, TIFF/PNG format and alignment, basic COCO category evidence, and declared validity. No mandatory reviewer, per-image approval or visual/AI field exists.
- The current policy holds the whole source when declared NoData, alpha, invalid masks or external raster sidecars are present. Crops are calculated for eligible sources and reject all positive source support from edge-connected exact-black pixels. White support is a diagnostic. Interior darkness, uniform teachers and canopy fraction do not gate eligibility. Photographic coverage remains unverified and the tool does not generate `valid_fraction`.
- `window_decision` reports eligibility, no clean window or technical error separately. Teacher pixels are the official nonzero RGB Tree+Canopy union, with area fraction reduced on the output grid and thresholded at 0.5. The report records the upstream label assumption once for the policy run. Semantic accuracy and a Panoptic-to-COCO segment crosswalk remain unverified.
- Use `final_decision` for adoption and `eligible_ids.train` as the only training allowlist. An eligible source with a HOLD window is omitted. `eligible_ids.val` is reserved for evaluation. Counts are derived from final decisions.

Identical frozen inputs, seed and rules are intended to reproduce decisions and crop coordinates. Record the Git commit and dirty flag, the running script's raw SHA-256, OS, Python, dependency versions and GDAL from the report. Use a clean checkout for a reviewed run. No cross-platform TIFF byte equality is required. This tool has synthetic test coverage. A separate A2 decision is required before a bounded real-data pilot can measure yield, runtime, memory and crop diversity.
