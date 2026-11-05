"""Local hash-verified mask overlay diagnostics for M1-2B."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from m1_2a_data_qa import labels


def overlay(rgb, mask):
    if rgb.shape[:2] != mask.shape:
        raise ValueError('Overlay grid mismatch')
    result = rgb.copy()
    result[mask == 1] = np.rint(0.65 * rgb[mask == 1] + 0.35 * np.array([255, 0, 255])).astype(np.uint8)
    return result



def decode_rle(rle):
    counts, shape = rle["counts"], rle["size"]
    if not isinstance(counts, list) or any(not isinstance(n, int) or n < 0 for n in counts):
        raise ValueError("Expected nonnegative uncompressed RLE counts")
    if sum(counts) != int(np.prod(shape)):
        raise ValueError("RLE length mismatch")
    flat = np.repeat(np.arange(len(counts)) % 2, counts).astype(np.uint8)
    return flat.reshape(shape, order="F")


def audit(cache, manifest):
    results = []
    for sample in manifest['samples']:
        folder = cache / str(sample['image_id'])
        for name, expected in sample['hashes'].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Original cache hash mismatch')
        rgb_image = Image.open(folder / 'rgb.bin')
        ann_image = Image.open(folder / 'annotation.bin')
        rgb, ann = np.array(rgb_image), np.array(ann_image)
        mask, qa = labels(rgb, ann, json.loads((folder / 'segments.json').read_text()))
        if not np.array_equal(mask, np.array(Image.open(folder / 'candidate-mask.png'))):
            raise ValueError('Regenerated mask differs')
        crowd_checks = []
        panoptic = ann.astype(np.uint32)
        ids = panoptic[:, :, 0] + 256 * panoptic[:, :, 1] + 65536 * panoptic[:, :, 2]
        segments = json.loads((folder / 'segments.json').read_text())
        for obj in json.loads((folder / 'coco_annotations.json').read_text()):
            if obj.get('iscrowd'):
                candidates = [x for x in segments if x['iscrowd'] and x['category_id'] == obj['category_id']]
                if len(candidates) != 1:
                    raise ValueError('Crowd correspondence is ambiguous')
                decoded = decode_rle(obj['segmentation'])
                mismatch = int(np.count_nonzero(decoded != (ids == candidates[0]['id'])))
                crowd_checks.append(dict(segment_id=candidates[0]['id'], pixels=int(decoded.sum()), mismatch=mismatch))
                if mismatch:
                    raise ValueError('Crowd RLE mismatch')
        combined = Image.new('RGB', (rgb_image.width * 2, rgb_image.height))
        combined.paste(rgb_image, (0, 0))
        combined.paste(Image.fromarray(overlay(rgb, mask)), (rgb_image.width, 0))
        combined.thumbnail((1600, 800))
        combined.save(folder / 'm1-2b-overlay.png')
        results.append({'image_id': sample['image_id'], 'pixel_qa': qa, 'crowd_rle_checks': crowd_checks,
                        'rgb_format': rgb_image.format, 'annotation_format': ann_image.format,
                        'rgb_orientation': rgb_image.getexif().get(274),
                        'annotation_orientation': ann_image.getexif().get(274),
                        'official_nonzero_mask_equal': bool(np.array_equal(mask, np.any(ann != 0, axis=-1))),
                        'overlay_sha256': hashlib.sha256((folder / 'm1-2b-overlay.png').read_bytes()).hexdigest()})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path('docs/implementation/M1_2A_SAMPLE_MANIFEST.json').read_text())
    print(json.dumps(audit(args.cache, manifest), indent=2))
