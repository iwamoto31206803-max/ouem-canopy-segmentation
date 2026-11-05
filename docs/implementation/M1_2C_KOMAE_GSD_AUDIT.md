# M1-2C Komae GSI imagery and GSD audit

## Sites and acquisition

Observed 2026-11-05 using official seamlessphoto z18. Candidate coordinates identify audit centers, with neighborhoods interpreted from the downloaded imagery. Geographic place-name boundaries were not independently surveyed.

| Site | Latitude | Longitude | Selection and observed scene |
| --- | --- | --- | --- |
| A Izumihoncho / city-hall vicinity | 35.634660 | 139.578200 | Dense urban housing, large public buildings, roads, rail/station vicinity and scattered crowns. Useful hard surfaces/shadow/tree contrast. |
| B Nogawa greenway vicinity | 35.642955 | 139.578622 | Housing and narrow curving tree-lined corridor beside a cemetery, with recreation grounds. Linear canopy/housing boundary candidate. |
| C Nishigawara Park / Tama river | 35.632894 | 139.566725 | Park canopy and paths adjacent to housing, sports grounds, embankment and river water. Grass and riverbank vegetation contrast. |

Site selection references include the [city park list](https://origin.city.komae.tokyo.jp/index.cfm/41%2C4600%2C313%2C3463%2Chtml) and [Tokyo metropolitan Komae facility coordinates](https://www.opendata.metro.tokyo.lg.jp/komae/132195_aed.pdf). The B representative coordinate is an audit candidate for the visible greenway vicinity, not a certified park entrance.

Each site used its containing XYZ tile plus its eight neighbors. Total 27 tiles, 515,352 downloaded payload bytes. Every tile returned HTTP 200, image/jpeg and JPEG decoding at 256 x 256. Missing tiles were not observed. Per-tile URLs, UTC timestamps, byte counts, hashes and coordinates are in the [manifest](M1_2C_KOMAE_TILE_MANIFEST.json). Originals, mosaics and GeoTIFFs remain in `C:\dev\ouem-data-cache\m1-2c-komae-01`.

## Grid and ground interval

Tile grid uses spherical Web Mercator R=6378137 m. At z18 the map interval is 2*pi*R/(256*2^18) = 0.5971642834779395 m/px. Each 3x3 mosaic has 768 x 768 pixels and spans 458.622169711 map metres per axis.

| Site | WGS84 geodesic east interval m | WGS84 geodesic north interval m | Spherical cosine estimate m |
| --- | --- | --- | --- |
| A | 0.485896732 | 0.483743171 | 0.485344357 |
| B | 0.485846521 | 0.483693629 | 0.485293982 |
| C | 0.485907417 | 0.483753717 | 0.485355080 |

The calculation projects each candidate center into EPSG:3857, offsets by one map pixel east and north separately, transforms endpoints back to lon/lat and measures WGS84 geodesic distances. Ellipsoidal distances differ slightly by direction. Grid spacing varies slightly across each footprint. These numbers describe the delivered tile sampling, while source camera/orthophoto resolving power remains unmeasured.

All three GeoTIFFs reopened as EPSG:3857, width=height=768, three uint8 RGB bands. Transform is (r,0,left,0,-r,top), with left/top from the northwest tile's outer edge. Bounds and full transforms are in the manifest. Pixel centers occur at half-pixel offsets from the outer edge. Tiles were pasted in increasing x and y with no resampling, reprojection or interpolation. GeoTIFF readback matched every decoded mosaic RGB pixel. JPEG decoding introduces the source compression already present in tiles.

The [GSI specification](https://maps.gsi.go.jp/development/siyou.html) describes the Japanese tile datum as JGD2011. EPSG:3857 is used here for the delivered web grid and WGS84 for local-distance approximation. No datum transformation or absolute survey accuracy claim is inferred from the numerical grid definition.

## Actual visual inspection

All three 768 x 768 RGB mosaics were displayed and inspected.

- A has visible crowns among roofs, paved roads and large buildings. Building shadows are substantial locally, especially near the rail/building corridor, but do not obscure the entire scene. Small garden and roadside vegetation is visible, with limited species/object separation at this sampling.
- B contains narrow connected greenery alongside housing and cemetery blocks. Small crowns and roof-edge shadows coexist. It supplies a constrained linear-greenery example rather than a large closed-canopy forest.
- C has the clearest concentrated park canopy, paths, open ground, embankment and river vegetation/water contrasts. Some dark tree shadows and sparse/leafless-looking crowns are present.

All have somewhat muted color and limited fine texture. No obvious tile-sized gaps, severe blur making all crowns unrecognizable, or conspicuous tile-edge color discontinuities were seen at native mosaic overview. Photogrammetric building lean and shadows can complicate boundaries. Quantitative blur, crown-label accuracy and subpixel orthorectification errors were not measured. Exact capture season/year and underlying survey source for each tile remain unknown. Visual foliage differences alone cannot establish season. No year-specific tile was acquired.

## Provisional training target and minimal next implementation

Recommend one initial nominal training GSD of 0.5 m/px for review. It is close to the observed delivered ground interval of approximately 0.484–0.486 m/px and provides a practical rounded operational scale. This recommendation is provisional pending implementation review. It does not derive original-photo native resolution from the tile zoom.

For the OAM-TCD subset, first establish whether its advertised 0.1m is nominal projected spacing or local ground spacing using each record CRS/bounds. The sampled records use EPSG:3395, whose metre spacing also needs scale correction. A nominal factor-5 transform is suitable only when explicitly recorded as a nominal 10cm-to-50cm recipe. Do not silently label it exact local-ground-GSD matching.

Next PR should freeze one shared crop/output grid, calculate output dimensions and effective spacing, reduce RGB using area/antialiasing, decode source panoptic IDs before aggregating binary canopy coverage, and threshold fractional coverage with an explicit initial value such as 0.5 for review. Preserve coverage fractions and valid-area fractions for QA. ID/category values must not be interpolated. Apply identical spatial crop/extent to RGB and labels, preserve uniform masks and record rounding, remainders and alignment. Source geometry/validity evidence remains a prerequisite for claiming true ground-scale matching.

## Usage, reproduction and checks

Use the [GSI content terms](https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html) and the layer-specific [official tile list](https://maps.gsi.go.jp/development/ichiran.html#seamlessphoto). This photo layer is listed under source-credit use. Credit the originals as 国土地理院 and derived mosaics as 国土地理院の地理院タイルを加工して作成, with the layer-list link. Third-party source conditions must be retained if identified. The listed special GRUS coverage is outside Komae.

```powershell
.\.venv\Scripts\python.exe -m pip install rasterio==1.4.3 pyproj==3.7.1
.\.venv\Scripts\python.exe scripts/m1_2c_gsi_audit.py --output '<fresh-external-directory>'
.\.venv\Scripts\python.exe -m pytest -q
```

Existing Python 3.11.9 venv was reused. Added rasterio 1.4.3 and pyproj 3.7.1 with their small runtime dependencies. pip check succeeded. Offline tests completed with 21 passed in 7.25 seconds. Tests cover XYZ placement, adjacent grid continuity and geodesic-vs-map spacing. Actual acquisition also validates each tile and complete GeoTIFF pixel readback. Limits are fixed to three sites, z18, nine tiles each, 1 MiB per tile and 32 MiB total response payload. A rejected chunk can overshoot the transfer threshold by one 64 KiB chunk. Errors produce an incomplete manifest and stop acquisition.
