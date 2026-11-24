# D6-A3-1 synthetic producer and reader

The CLI is restricted to synthetic fixtures. It requires a synthetic plan, a cache marker containing `D6-A3-1 SYNTHETIC FIXTURE` plus a newline, and fixture scenes no larger than 512×512. A3-2 real imagery requires separate review and authorization. Keep all synthetic inputs and outputs outside the Git tree. The tests construct the full contract and fixtures.

```powershell
python -B scripts/d6a3_produce.py --plan C:/synthetic/inputs/plan.json `
  --plan-sha256 <reviewed-SHA256> --root C:/synthetic/inputs/cache `
  --evidence C:/synthetic/inputs/stac --output C:/synthetic/generation
```

The pinned plan uses `schema: d6-research-plan-v1`, `mode: synthetic`, `windows_per_source: {train: 1, val: 1}`, ordered `source_ids`, and explicit `allow_authenticity`. It references `a1_contract`, `a1_report` and `protected` by relative path and SHA-256. The protected index pins the historical v5, historic Val18 and GT54 metadata sources and their normalized OAM aliases and WGS84 footprints. This synthetic producer consumes a trusted, externally pinned normalized index. It does not derive or verify completeness against original historical manifests; that provenance remains an A3-2 review gate. Official Test catalog metadata is checked alongside Train. No holdout pixels enter the producer.

The producer uses the exact A1-selected scene-global crop and reference area reduction. It writes RGB, Float32 canopy fraction and binary mask derived by reopening the fraction and applying >=0.5. Product descriptions are relative to each source directory and record byte size and SHA-256. No validity TIFF or new label review is created. The reader uses the committed full-grid slice key but returns already materialized 224×224 products.

## Resource and restart contract

The synthetic plan must explicitly provide positive `commit_bytes`, `rss_bytes`, `host_available_bytes`, `seconds`, `output_bytes` and `temporary_bytes`. The Windows CLI runs a single source worker in a verified Job with a commit cap and one active process. The parent checks the actual peak working set and a monotonic deadline and kills and waits on failure. Peak observation can detect an RSS overshoot after it occurs. The hard Job commit limit constrains memory allocation. Failure to observe memory or install the Job blocks publication. Worker timings are printed as a separate execution result rather than hashed into the dataset.

Each compressed TIFF is encoded in a GDAL MemoryFile and accepted for disk write only if its byte length fits the remaining reservation. The single writer accounts for existing published and staged payloads, plan, source pins and receipts. The cap covers logical payload bytes. The free-space check provides headroom for filesystem overhead, which is not a measured quota. Plan/receipt scans and per-window verification by the coordinator occur outside the source-worker Job. Actual full-scene RAM, I/O and power-loss durability are unverified.

- **STAGED** — `sources/.stage-<source_key>` may contain partial products or a source receipt. Without a coordinator commit pin, resume stops. Inspect and quarantine the stage outside the generation before a freshly authorized attempt. The CLI never deletes a stage.
- **VERIFIED** — `commits/<source_key>.json` pins the verified receipt after successful worker completion, actual resource checks and independent product readback. Resume verifies the staged files again.
- **PUBLISHED** — the stage moves to `sources/<source_key>` on the same volume. Verified published sources can be skipped on `--resume`. A missing or corrupt committed source stops and requires a separate repair generation.
- **FROZEN** — a flushed canonical manifest is published last using an exclusive hardlink. The caller supplies its SHA-256 to `D6Data`. The reader requires a complete exact inventory and matching source/manifest pins. A leftover claim from forced termination blocks resume until independently inspected. Use a local Windows filesystem supporting no-replace rename and hardlinks.

The CLI needs no model weights, package download, cloud mount or CUDA. Review changes to the admission, protected-source, resource or crop policy separately.

## A3-2 approval preflight

The optional pair `--real-approval <reviewed-token.json>` and `--real-approval-sha256 <independent-pin>` checks a separately pinned `d6-real-pilot-plan-v1` and `d6-real-pilot-approval-v1` token. The token binds the exact plan SHA and one bounded real-output pilot. This command path is restricted to metadata verification. It stops before raster opens, staging or output, because a physical volume quota and whole-attempt coordinator guard have yet to be demonstrated. The ordinary CLI remains synthetic-only.

The proposal binds two A1 report/contract pairs, official Train/Test/COCO hashes, historical v5 freeze/source index, STAC pins, exact source identities and original crops, a two-per-role cap, both authenticity classes under an explicit policy, and positive resource limits. Same-role historical Train reuse is tracked. Historic Val18 and official Test are protected. GT54 is deferred and geographic independence remains unverified. The approved stage generated A1 recipes for six fresh Val sources; no real output token was issued.

For a future reviewer-approved implementation, use a dedicated, same-volume local NTFS staging/output root with a demonstrable 64 MiB physical cap. A provisioned size-bounded volume or an existing scoped enforced quota must be tested for allocated bytes, headroom and overflow on the intended volume. Retain MemoryFile logical reservations and explicit 4 MiB per stage / 16 MiB total logical ceilings. The proposed limits also require 6 GiB host available, at least 256 MiB free underlying headroom, 2.5 GiB worker commit, 2 GiB sampled-peak RSS cutoff, 120 s/source and a 600 s attempt cap. Reuse the Windows Job worker guard and add coordinator deadline/RAM coverage before enabling execution. Confirm same-volume hardlink/rename, junction rejection, kill/wait/reap and restart accounting with synthetic or fake-real inputs first. No host configuration change or real file generation was performed here.
