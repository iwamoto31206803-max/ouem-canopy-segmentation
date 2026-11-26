# D6-A5 cached OAM-TCD inventory and research generation

[Issue #118](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/118) authorized one metadata-only acquisition and CPU data campaign from merged A4. The existing imagery and annotation cache, fixed v5 and Val18, and A4 patch products were read-only inputs. The unified D6 generation references A4 in place with its original roster and manifest pins.

## Reusable inventory

The local inventory contains one JSONL row per OAM group, with official roles/folds, source-object alias digests, known grid envelopes, tile identifiers and paired-file presence, STAC SHA and lookup status, A1 outcomes, and materialized image identity. The original paired-file check used filenames. Tile-level QA and selected-byte SHA checks run later. Catalog Test metadata contributes to independence checks and no Test pixels were inspected.

| Catalog scope | OAM groups | Tiles | Cached RGB/annotation pairs |
| --- | ---: | ---: | ---: |
| Official Train, folds 1–4 | 384 | 3,341 | 3,341 |
| Official Train, fold 0 (Val) | 102 | 828 | 828 |
| Official Test | 55 | 439 | Not checked |
| Total metadata | 541 | 4,608 | 4,169 official Train pairs |

The catalogs supplied 538 distinct normalized object aliases. Three groups lacked an alias and no alias spanned multiple OAM groups. The 541 groups span the two pinned official catalogs, rather than a claim about all upstream OpenAerialMap sources.

To rebuild privately, supply the SHA-256 of a retained config JSON with `cache`, `catalog_sha256` (official Train and Test keys), pinned `protected`, `parent_roster` and `a1_reports` references. File references carry local absolute `path` and `sha256`. Set the roster to the latest completed generation and include each retained A1 report. Use a fresh output path and the cached evidence directory. Optional `--fetch-log` reads the existing receipt from the metadata campaign.

```powershell
.\.venv\Scripts\python.exe -B scripts/d6a5_inventory.py `
  --config "<private config>" --config-sha256 "<retained SHA>" `
  --evidence "<local STAC cache>" --fetch-log "<existing lookup receipt>" `
  --out "<new private inventory.jsonl>"
```

The default is offline. A separately authorized metadata refresh uses `--lookups` and a fresh `--fetch-log` with known eligible missing groups, fresh Val first, then unused Train and seeded region rotation. The current implementation uses one public official endpoint, a 12-second socket timeout, roughly one second between serial requests, 128 KiB accepted body cap per response, with a one-byte oversize probe, no retries or redirects, a 500-lookup maximum and a stop on rate limits or systemic failures. Snapshots are exclusive writes with an actual SHA. Local eligibility still checks OAM identity and the existing rights and known-envelope rules.

## STAC and candidate results

| Coverage | Train groups | Val groups | Total |
| --- | ---: | ---: | ---: |
| Cached before | 96 | 26 | 122 |
| Cached after | 331 | 83 | 414 |
| Relevant lookups still missing after attempt | 19 | 4 | 23 |

315 missing groups were looked up once. There were 292 usable snapshots, 19 rejected length mismatches, three HTTP 404s, and one connection reset. The log accounted for 987,128 response-body bytes read, excluding HTTP/TLS overhead and any unobserved bytes from the reset. The first and last request starts were 506.9 seconds apart. No source asset or annotation download was performed.

Three representative prior bbox exclusions were checked using the catalog EPSG:3395 bounds transformed to WGS84. Both transformation paths agreed. Two tiles extended north of the STAC declaration and one extended west and north. This sample identifies real disagreements between the declared envelopes. The reason for the declarations and actual photographed coverage remain unresolved. Geometry rules were retained.

A5 froze at most one candidate per source identity before pixel QA, with seed 118, the existing 15° geographic rotation, fresh Val, and new-to-v5 Train priority. Prior HOLD tiles were skipped without changing the rules. The post-fetch inventory had 169 Train and 43 Val documentary-eligible groups outside the A4 parent. The frozen candidate list comprised 169 Train and 42 Val. A1 produced 164 eligible (135 Train / 29 Val), 46 HOLD for no clean 224px window and one technical encoding failure. The rejected inputs and all existing labels were left as-is.

The new roster preserves K=1, each exact A1 crop, and the existing Float32 fraction and 0.5 mask threshold. The parent binding pins the whole previous A4 aggregate, which pins its original roster and shard manifest digests. The composed Reader checks the exact parent prefix and each new roster slice, global image/group/alias/patch uniqueness, roles, crop keys and product bytes. Historical v5 and its fixed Val18 remain separate; the v5 compatibility reference is 70 Train originals with 139 windows and 18 fixed Val windows.

## Generation and verification result

**DATASET_READER_PASS.** A5 reached 200 source groups, comprising 175 Train and 25 fresh Val. The exact 80-source A4 generation was referenced in place. A5 generated 120 new patches (100 Train and 20 Val), all from groups new to v5. The combined generation includes 56 same-role historical v5 Train groups and 144 groups new to v5 (119 Train and 25 Val). Groups with externally expected input SHA total two, with locally observed SHA for the other 198. Region counts at the 15° catalog-center scale are 62 Train and 20 Val. The frozen eligible reserve is 35 Train and nine Val outside the 200-group generation.

| Windows CPU phase | Sources | Wall seconds | Peak RSS bytes | Peak commit bytes | Logical output bytes | Observed allocation bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 QA | 211 | 204.35 | 839,778,304 | 844,140,544 | — | — |
| Shard 01 | 35 | 63.00 | 802,643,968 | 781,193,216 | 5,084,810 | 5,378,256 |
| Shard 02 | 29 | 51.57 | 802,271,232 | 780,509,184 | 4,138,048 | 4,406,064 |
| Shard 03 | 28 | 50.36 | 802,906,112 | 781,479,936 | 4,004,261 | 4,270,656 |
| Shard 04 | 28 | 49.90 | 803,319,808 | 781,733,888 | 4,024,129 | 4,254,272 |

The serial production CLIs took 214.84 seconds total. Every shard completed with exit 0 and no retry. The first shard used the measured A4 guard basis, with 900 seconds, 24 MiB logical and 96 MiB sampled allocation limits. The other deadlines were 600 seconds, with 20–21 MiB logical and 80–84 MiB sampled allocation limits. Per-source staging remained capped at 4 MiB and 120 seconds. The one-process Windows Job retained the 2,560 MiB commit cap, 2 GiB RSS stop and 256 MiB free-space floor. Observed allocation is sampled per shard.

D6Shards reopened all 600 GeoTIFFs and all 200 RGB/target tensors. It verified product pins, actual grid/crop, dtypes, ranges and the saved Float32 fraction threshold. One CPU `make_plan` cycle used all Train patches once and the fixed crop in each Val group. The integrated readback took 11.59 seconds. The referenced outputs occupy 28,666,601 logical bytes and cover 10,035,200 patch pixels. Train mean canopy fraction is 29.57%, with nine zero-canopy and 13 nonzero below 1%; Val mean is 35.27%, with one zero-canopy and no nonzero below 1%. Both splits contain a fully positive patch.

Use `D6Shards(manifest_path, manifest_sha256)` with the retained aggregate and the parent/shards at their bound local paths. Keep the latest inventory config, JSONL, lookup log and exact input/output pins privately with the campaign receipts. The latest inventory was rebuilt offline against the completed generation and its retained A1 evidence. Its 541 rows comprise 200 materialized groups, 44 additional A1-eligible, 48 documentary-eligible with per-tile QA state, 23 relevant groups still missing STAC, 171 HOLD, and 55 Test metadata-only groups. Portable cloud path rebinding remains deferred to D6-B.

- Executed QA and production code `910f562742f0780c95e33105a03daa548254bef7`
- Generation ID `9571151c0037d2ba7b05c9ad46a860579efa547d62a7248eccac7c96b47faf60`
- Manifest SHA-256 `66b2ab066eb362c583456bc270609dca533da2bafbf8f9645c07529409e4cff5`
- Roster SHA-256 `6989821353b5028ae51663ffb5be813c531cf39655de9dcca0164fab367193a9`
- Current private inventory SHA-256 `5f29e2600154cdf59beac4dc67214972fde618acefa8fdd75b0e1b4fdbcb3dd0`

Windows verification passed 26 targeted tests and 783 full-suite tests, with two existing symlink privilege skips and zero failures. Pending-deprecation notices were filtered in the full run; the expected duplicate ZIP entry warning remained. New tests cover the composed parent binding and full synthetic patch readback, identity and split misuse, missing/corrupt references and the no-retry metadata transport. Private input and output evidence was reconciled locally. Formal R1 remains HOLD for all 150, GT54 independence is unverified and deferred, and no GPU, model training or performance evaluation was performed.
