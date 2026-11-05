"""Explicit opt-in official Satlas CPU smoke. No checkpoint downloads."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
import re
import sys
import time


def checkpoint_digest(path, expected):
    if not expected or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("An expected 64-digit SHA-256 is required")
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError("Checkpoint SHA-256 mismatch")
    return actual


def synthetic(torch, size, seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (1, 3, size, size), generator=generator).float() / 255
    mixed = torch.randint(0, 2, (1, size, size), generator=generator)
    return images, {"mixed": mixed, "background": torch.zeros_like(mixed),
                    "canopy": torch.ones_like(mixed)}


def validate_output(torch, probabilities, loss, size):
    if tuple(probabilities.shape) != (1, 2, size, size):
        raise ValueError(f"Unexpected probability shape {tuple(probabilities.shape)}")
    if not torch.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid probabilities")
    if not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("Class probabilities do not sum to one")
    if loss is None or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Invalid official loss")


def gradient_evidence(torch, component):
    names = []
    for name, parameter in component.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise ValueError(f"Nonfinite gradient in {name}")
            if torch.count_nonzero(parameter.grad):
                names.append(name)
    if not names:
        raise ValueError("No finite nonzero component gradients")
    return {"finite_nonzero_parameters": len(names), "example": names[0]}


def run(args, record):
    import torch
    import satlaspretrain_models as satlas
    import satlaspretrain_models.model as official_model
    import satlaspretrain_models.models.heads as heads
    import satlaspretrain_models.models.fpn as fpn
    import satlaspretrain_models.models.backbones as backbones
    import satlaspretrain_models.utils as utils

    record["versions"] = {name: version(name) for name in
                          ("torch", "torchvision", "satlaspretrain-models", "pytest")}
    record["official_source_sha256"] = {
        name: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
        for name, module in (("model", official_model), ("heads", heads), ("fpn", fpn),
                             ("backbones", backbones), ("utils", utils))}
    record["torch_cuda_build"] = torch.version.cuda
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    weights = None
    if args.init == "checkpoint":
        if args.checkpoint is None:
            raise ValueError("--checkpoint and --sha256 are required in checkpoint mode")
        digest = checkpoint_digest(args.checkpoint, args.sha256)
        record["checkpoint"] = {"filename": args.checkpoint.name, "sha256": digest,
                                "source": args.checkpoint_source or "NOT_RECORDED"}
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Expected official nonempty tensor state dict")
        if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in weights.items()):
            raise ValueError("Expected a flat tensor state dict")
    model = satlas.Model(num_channels=3, multi_image=False, backbone=satlas.Backbone.SWINB,
                         fpn=True, head=satlas.Head.SEGMENT, num_categories=2, weights=weights).cpu()
    # Official Model performs strict per-component load_state_dict calls.
    record["components"] = {"backbone": "strict_loaded" if weights is not None else "random",
                            "fpn": "strict_loaded" if weights is not None else "random",
                            "upsample": "random", "head": "random"}
    del weights
    model.requires_grad_(True)
    model.train()
    images, masks = synthetic(torch, args.size, args.seed)
    record["input_shape"] = list(images.shape)
    record["target_shape"] = list(masks["mixed"].shape)
    record["cases"] = {}
    for name, target in masks.items():
        started = time.perf_counter()
        model.zero_grad(set_to_none=True)
        probabilities, loss = model(images, target)
        validate_output(torch, probabilities, loss, args.size)
        loss.backward()
        record["cases"][name] = {
            "output_shape": list(probabilities.shape), "loss": loss.item(),
            "gradients": {component: gradient_evidence(torch, getattr(model, component))
                          for component in ("head", "fpn", "backbone")},
            "seconds": time.perf_counter() - started}
        del probabilities, loss
    record["status"] = "STRUCTURAL_PASS" if args.init == "random" else "PRETRAINED_PASS"
    record["pretrained_status"] = "PRETRAINED_NOT_VERIFIED" if args.init == "random" else "PRETRAINED_PASS"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=("random", "checkpoint"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--checkpoint-source", help="Publisher URL and immutable revision, if known")
    parser.add_argument("--output", type=Path, help="Optional JSON result destination")
    args = parser.parse_args(argv)
    record = {"mode": args.init, "device": args.device, "size": args.size, "seed": args.seed,
              "threads": args.threads, "python": platform.python_version(),
              "platform": platform.platform(), "utc": datetime.now(timezone.utc).isoformat(),
              "status": "NOT_VERIFIED", "pretrained_status": "PRETRAINED_NOT_VERIFIED"}
    started = time.perf_counter()
    code = 0
    try:
        if args.size < 1 or args.threads < 1:
            raise ValueError("Size and threads must be positive")
        if args.init == "random" and (args.checkpoint or args.sha256 or args.checkpoint_source):
            raise ValueError("Checkpoint arguments require checkpoint mode")
        run(args, record)
    except (ImportError, FileNotFoundError, MemoryError) as error:
        record["exception"] = f"{type(error).__name__}: {error}"
        code = 2
    except Exception as error:
        record["status"] = "FAIL"
        record["exception"] = f"{type(error).__name__}: {error}"
        code = 1
    record["total_seconds"] = time.perf_counter() - started
    result = json.dumps(record, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
