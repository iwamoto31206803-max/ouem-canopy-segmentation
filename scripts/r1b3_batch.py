"""Small, fail-closed operator CLI for R1b-3 batches 02-05.

This is not an approval issuer. prepare only writes an unapproved candidate.
No network access, Campaign initialization, retries or multi-batch execution.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import r1b_campaign as ledger
import r1b_qa_runner as qa
import r1b3_scope as scope

AUDITS = Path(os.environ.get("OUEM_R1B3_AUDITS", r"C:\dev\ouem-data-audits"))
SCOPE_SHA = "b4345e73da9931b15ef473c57312189a173bb0e61dbb282aba65ac2ac03b8999"
CONTRACT_SHA = "be5db1d87c795988b93b84b3b4fb6476e6e15e2b7a48160e25205e66b01042e9"
TEMPLATE_SHA = "c137cb948879b5a1d4c954540ecf7d58661ffb78299fb79b0572d5ebdef93032"
SCOPES = ["qa_local", "qa_preview", "qa_report"]


def require(ok, reason):
    if not ok:
        raise ledger.Stop("BATCH_CLI_STOP: " + reason)


def load(path, expected_sha):
    path = Path(path)
    ledger.ordinary(path)
    require(ledger.sha(path) == expected_sha, "file SHA mismatch: " + str(path))
    return qa.strict_json(path.read_bytes())


def batch_name(number):
    return f"r1b3-batch{number:02d}-local-01"


def files_and_samples(contract, batch):
    """Read file hashes, never decode imagery or modify source files."""
    cache = Path(contract["cache_root"])
    sample_index = {r["image_id"]: r for r in contract["_scope_samples"]}
    pins = {}
    samples = {}
    for image_id in batch:
        sample = sample_index[image_id]
        source = sample["source_files"]
        require(source == contract["source_files"][str(image_id)],
                "contract/scope filename mismatch")
        for kind in ("rgb", "annotation"):
            name = source[kind]
            require(isinstance(name, str) and "\\" not in name and not name.startswith("/"),
                    "unsafe source path")
            path = cache / name
            ledger.ordinary(path)
            require(path.resolve().is_relative_to(cache.resolve()), "source escapes cache")
            require(path.is_file(), "source missing: " + name)
            pins[name] = ledger.sha(path)
        samples[str(image_id)] = dict(
            oam_id=sample["oam_id"], split=sample["split"],
            validation_fold=sample["validation_fold"])
    require(len(pins) == 2 * len(batch), "duplicate source filenames")
    return pins, samples


def context(args):
    """Validate the immutable scope, old evidence and current Campaign in read-only mode."""
    scope_path = Path(args.scope)
    ledger.ordinary(scope_path)
    view = scope.ScopeCampaign.inspect(args.root, args.registry, scope_path, args.scope_sha256)
    require(view["status"] == "CLEAN_PAUSED", "Campaign not CLEAN_PAUSED")
    contract = view["contract"]
    require(ledger.sha(Path(args.root) / "contract.json") == args.contract_sha256,
            "contract SHA")
    require(contract["schema"] == scope.CAMPAIGN_SCHEMA and
            contract["plan_sha256"] == args.scope_sha256,
            "campaign scope/schema mismatch")
    require(contract["cache_root"] == str(Path(args.cache).resolve()), "cache root mismatch")
    require(contract["limits"]["network_bytes"] == 0 and
            contract["limits"]["requests"] == 0 and
            contract["limits"]["storage_bytes"] <= 256 * 1024**2,
            "offline budget contract")
    records = load(scope_path, args.scope_sha256)
    require(records["scope_id"] == contract["scope_id"], "scope ID")
    require(len(records["samples"]) == 125 and len(contract["targets"]) == 125,
            "125 image scope")
    require(len(contract["batches"]) == 5, "fixed batches")
    batch = contract["batches"][args.batch - 1]
    require(len(batch) == len(set(batch)) == 25, "batch not 25 unique IDs")
    require(sum(contract["targets"][str(i)] == "train" for i in batch) == 20 and
            sum(contract["targets"][str(i)] == "val" for i in batch) == 5,
            "batch split quota")
    ids = [r["image_id"] for r in records["samples"]]
    require(ids == [i for group in contract["batches"] for i in group],
            "batch/ordered scope mismatch")
    require(all(str(i) in contract["targets"] for i in batch), "unplanned image")
    ops = view["operations"]
    require(not any(f"qa-{kind}-{i}" in ops for i in batch
                    for kind in ("preview", "report")), "image already processed")
    run_id = batch_name(args.batch)
    require(not any(f"qa-{kind}-{run_id}" in ops for kind in ("run", "summary")),
            "run already used")
    require(view["usage"]["unresolved_storage_reservation_bytes"] == 0 and
            not view["usage"]["unresolved_operations"] and
            view["usage"]["requests"] == 0 and
            view["usage"]["network_charged_bytes"] == 0,
            "unresolved or network usage")
    require(view["usage"]["storage_including_reservations_bytes"] <
            contract["limits"]["storage_bytes"], "storage budget exhausted")
    env = qa.environment()
    require(env["git_dirty"] is False, "Git working tree not clean")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=qa.REPO, text=True).strip()
    require(branch == "main", "must run from main branch")
    # The scope is reconstructed during inspect, including old Campaign receipts.
    contract["_scope_samples"] = records["samples"]  # transient, never persisted
    return view, contract, batch, env


def prepare(args, contract, batch, env):
    template = load(args.template, args.template_sha256)
    require(template["schema"] == "ouem-r1b-qa-approval-v2" and
            template["approved"] is False and
            template["reviewer_ref"] == "PENDING_INDEPENDENT_REVIEW",
            "template must be unapproved")
    require(template["contract_sha256"] == args.contract_sha256 and
            template["plan_sha256"] == args.scope_sha256 and
            template["scope_sha256"] == args.scope_sha256 and
            template["scope_id"] == contract["scope_id"] and
            template["targets"] == contract["targets"] and
            template["limits"] == contract["limits"], "template contract/scope")
    require(template["mode"] == "production_qa" and
            template["scopes"] == SCOPES and
            template["local_stac"] == {}, "template scopes/mode")
    pins, samples = files_and_samples(contract, batch)
    coco = Path(contract["cache_root"]) / "archives/dataset/holdout/train.json"
    ledger.ordinary(coco)
    require(ledger.sha(coco) == template["coco_sha256"], "COCO SHA")
    candidate = copy.deepcopy(template)
    name = batch_name(args.batch)
    candidate.update(
        approval_id=name + "-candidate-v1",
        reviewer_ref="PENDING_INDEPENDENT_REVIEW",
        approved=False,
        allowed_image_ids=batch,
        environment=env,
        candidate_created_utc=datetime.now(timezone.utc).isoformat())
    candidate["input_pins"]["source_sha256"] = pins
    candidate["input_pins"]["samples"] = samples
    candidate["execution_intent"] = dict(
        run_id=name, ids=batch, fetch_missing_stac=False,
        additional_http_bytes=0, r1_acceptance="HOLD",
        authorization_status="PENDING_INDEPENDENT_REVIEW")
    destination = Path(args.output)
    ledger.ordinary(destination)
    for protected in (args.root, args.registry, args.scope, args.cache, qa.REPO):
        protected = Path(protected).resolve()
        require(destination.resolve() != protected and
                protected not in destination.resolve().parents,
                "candidate path overlaps protected inputs")
    require(destination.parent.is_dir(), "candidate parent directory missing")
    ledger.durable(destination, ledger.encoded(candidate))  # exclusive no overwrite
    print(json.dumps(dict(status="UNAPPROVED_CANDIDATE_CREATED",
        batch=args.batch, images=len(batch), train=20, val=5,
        file=str(destination), sha256=ledger.sha(destination),
        source_sha256_pins=len(pins), approved=False, qa_executed=False),
        ensure_ascii=False, indent=2))


def verify_approved(args, contract, batch, env):
    """Bind reviewed candidate, approved copy, all 50 source hashes and environment."""
    require(args.approval is not None and args.approval_sha256 and
            args.candidate_sha256, "approval and both reviewed SHA-256 values required")
    candidate = load(args.candidate, args.candidate_sha256)
    approved = load(args.approval, args.approval_sha256)
    require(candidate["approved"] is False and
            candidate["reviewer_ref"] == "PENDING_INDEPENDENT_REVIEW",
            "candidate no longer unapproved")
    normalized = copy.deepcopy(approved)
    require(approved["approved"] is True and
            isinstance(approved.get("reviewer_ref"), str) and
            approved["reviewer_ref"] not in ("", "PENDING_INDEPENDENT_REVIEW") and
            approved["approval_id"] == batch_name(args.batch) + "-approved-v1" and
            approved["execution_intent"]["authorization_status"] ==
            f"APPROVED_FOR_BATCH{args.batch:02d}_LOCAL_QA",
            "approved copy missing external review reference")
    normalized["approved"] = False
    normalized["approval_id"] = candidate["approval_id"]
    normalized["reviewer_ref"] = candidate["reviewer_ref"]
    normalized["execution_intent"]["authorization_status"] = (
        candidate["execution_intent"]["authorization_status"])
    require(normalized == candidate, "approval differs from candidate beyond four fields")
    require(approved["allowed_image_ids"] == batch and
            approved["execution_intent"]["ids"] == batch and
            approved["execution_intent"]["run_id"] == batch_name(args.batch) and
            approved["scopes"] == SCOPES and approved["local_stac"] == {} and
            approved["contract_sha256"] == args.contract_sha256 and
            approved["scope_sha256"] == args.scope_sha256 and
            approved["scope_id"] == contract["scope_id"] and
            approved["targets"] == contract["targets"] and
            approved["limits"] == contract["limits"] and
            approved["environment"] == env and
            approved["execution_intent"]["fetch_missing_stac"] is False and
            approved["execution_intent"]["additional_http_bytes"] == 0,
            "approved copy bindings")
    require(approved["plan_id"] == contract["plan_id"] and
            approved["plan_sha256"] == contract["plan_sha256"] and
            approved["campaign_id"] == contract["campaign_id"] and
            approved["cache_root"] == contract["cache_root"] and
            approved["output_root"] == str(Path(args.root).resolve()),
            "approved campaign binding")
    pins, samples = files_and_samples(contract, batch)
    require(approved["input_pins"]["source_sha256"] == pins and
            approved["input_pins"]["samples"] == samples,
            "source SHA pins differ from current originals")
    coco = Path(contract["cache_root"]) / "archives/dataset/holdout/train.json"
    ledger.ordinary(coco)
    require(ledger.sha(coco) == approved["coco_sha256"], "COCO SHA")
    return approved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "prepare", "check", "execute"))
    parser.add_argument("--batch", type=int, choices=(2, 3, 4, 5), required=True)
    parser.add_argument("--root", type=Path, default=AUDITS / "r1b3-pool150-campaign-01")
    parser.add_argument("--registry", type=Path, default=AUDITS / "r1b3-permanent-registry")
    parser.add_argument("--scope", type=Path, default=AUDITS / "r1b3-pool150-scope-01.json")
    parser.add_argument("--scope-sha256", default=SCOPE_SHA)
    parser.add_argument("--contract-sha256", default=CONTRACT_SHA)
    parser.add_argument("--cache", type=Path, default=Path(os.environ.get(
        "OUEM_R1B3_CACHE", r"C:\dev\ouem-data-cache\oam-tcd-zenodo-v1")))
    parser.add_argument("--template", type=Path,
                        default=AUDITS / "r1b3-batch01-local-01-approval-candidate.json")
    parser.add_argument("--template-sha256", default=TEMPLATE_SHA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(args.action == "execute" or not args.execute,
            "--execute is allowed only for execute action")
    require(args.action != "execute" or args.execute,
            "execute action requires explicit --execute flag")
    args.output = args.output or AUDITS / (batch_name(args.batch) + "-approval-candidate.json")
    args.candidate = args.candidate or args.output
    view, contract, batch, env = context(args)
    if args.action == "inspect":
        print(json.dumps(dict(status=view["status"], batch=args.batch,
            ids=batch, completed_operations=len(view["operations"]),
            usage=view["usage"], contract_sha256=args.contract_sha256,
            qa_executed=False), ensure_ascii=False, indent=2))
        return 0
    if args.action == "prepare":
        prepare(args, contract, batch, env)
        return 0
    verify_approved(args, contract, batch, env)
    runner_args = [
        "check-approval" if args.action == "check" else "run",
        "--root", str(args.root), "--registry", str(args.registry),
        "--plan", str(args.scope), "--plan-sha256", args.scope_sha256,
        "--approval", str(args.approval),
        "--approval-sha256", args.approval_sha256,
        "--ids", *(str(i) for i in batch)]
    if args.action == "execute":
        runner_args += ["--execute", "--run-id", batch_name(args.batch)]
    return qa.main(runner_args, campaign_type=scope.ScopeCampaign)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ledger.Stop, OSError, KeyError, ValueError, TypeError) as exc:
        print("STOP:", exc, file=__import__("sys").stderr)
        raise SystemExit(1)
