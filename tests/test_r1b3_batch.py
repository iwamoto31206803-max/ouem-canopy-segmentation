"""Offline tests for the small R1b3 operator CLI; no production paths used."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import r1b_campaign as ledger
import r1b3_batch as cli


@pytest.fixture
def prepared(tmp_path):
    cache = tmp_path / "cache"
    coco = cache / "archives/dataset/holdout/train.json"
    coco.parent.mkdir(parents=True)
    coco.write_text('{"annotations":[]}', encoding="utf-8")
    targets = {str(i): "val" if i % 5 == 4 else "train" for i in range(25, 150)}
    ids = list(range(50, 75))
    sample_rows = []
    source_files = {}
    for i in ids:
        stem = f"source-{i}_{i}"
        files = dict(
            rgb=f"extracted/dataset/holdout/images/{stem}.tif",
            annotation=f"extracted/dataset/holdout/masks/{stem}.png",
            byte_validation="NOT_VERIFIED_R1B2B_REQUIRED")
        source_files[str(i)] = files
        sample_rows.append(dict(
            image_id=i, oam_id=f"source-{i}", split=targets[str(i)],
            validation_fold=0 if targets[str(i)] == "val" else 1,
            source_files=files))
        for key in ("rgb", "annotation"):
            path = cache / files[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic-{key}-{i}".encode())
    limits = dict(network_bytes=0, requests=0, response_bytes=65536,
                  storage_bytes=256 * 1024**2)
    contract = dict(
        _scope_samples=sample_rows, source_files=source_files,
        cache_root=str(cache.resolve()), targets=targets, limits=limits,
        scope_id="ouem-r1b3-test", plan_id="ouem-r1b3-test",
        plan_sha256="a" * 64, campaign_id="synthetic")
    args = SimpleNamespace(
        batch=2, contract_sha256="b" * 64, scope_sha256="a" * 64,
        scope=tmp_path / "scope.json", root=tmp_path / "campaign",
        registry=tmp_path / "registry", cache=cache,
        template=tmp_path / "template.json", template_sha256=None,
        output=tmp_path / "batch02-candidate.json",
        candidate=tmp_path / "batch02-candidate.json",
        candidate_sha256=None, approval=tmp_path / "batch02-approved.json",
        approval_sha256=None)
    template = dict(
        schema="ouem-r1b-qa-approval-v2", approved=False,
        approval_id="old-candidate", reviewer_ref="PENDING_INDEPENDENT_REVIEW",
        mode="production_qa", scopes=list(cli.SCOPES), local_stac={},
        contract_sha256=args.contract_sha256,
        plan_sha256=args.scope_sha256, scope_sha256=args.scope_sha256,
        scope_id=contract["scope_id"], targets=targets, limits=limits,
        coco_sha256=ledger.sha(coco),
        input_pins=dict(source_sha256={}, samples={}),
        execution_intent=dict(authorization_status="PENDING_INDEPENDENT_REVIEW"),
        output_root=str(args.root.resolve()), cache_root=str(cache.resolve()),
        plan_id=contract["plan_id"], campaign_id=contract["campaign_id"])
    args.template.write_bytes(ledger.encoded(template))
    args.template_sha256 = ledger.sha(args.template)
    env = dict(git_sha="synthetic", git_dirty=False, runner_sha256="synthetic")
    return args, contract, ids, env


def test_prepare_only_writes_unapproved_candidate(prepared):
    args, contract, ids, env = prepared
    cli.prepare(args, contract, ids, env)
    candidate = cli.load(args.output, ledger.sha(args.output))
    assert candidate["approved"] is False
    assert candidate["allowed_image_ids"] == ids
    assert candidate["environment"] == env
    assert candidate["execution_intent"]["run_id"] == "r1b3-batch02-local-01"
    assert candidate["execution_intent"]["fetch_missing_stac"] is False
    assert len(candidate["input_pins"]["source_sha256"]) == 50
    assert len(candidate["input_pins"]["samples"]) == 25
    with pytest.raises(FileExistsError):
        cli.prepare(args, contract, ids, env)


def test_explicit_review_copy_only_four_fields(prepared):
    args, contract, ids, env = prepared
    cli.prepare(args, contract, ids, env)
    args.candidate_sha256 = ledger.sha(args.candidate)
    approved = json.loads(args.candidate.read_bytes())
    approved["approved"] = True
    approved["approval_id"] = "r1b3-batch02-local-01-approved-v1"
    approved["reviewer_ref"] = "independent-review-issue-xyz"
    approved["execution_intent"]["authorization_status"] = "APPROVED_FOR_BATCH02_LOCAL_QA"
    args.approval.write_bytes(ledger.encoded(approved))
    args.approval_sha256 = ledger.sha(args.approval)
    assert cli.verify_approved(args, contract, ids, env)["approved"] is True
    tampered = copy.deepcopy(approved)
    tampered["input_pins"]["samples"][str(ids[0])]["oam_id"] = "substituted"
    args.approval.write_bytes(ledger.encoded(tampered))
    args.approval_sha256 = ledger.sha(args.approval)
    with pytest.raises(ledger.Stop, match="beyond four fields"):
        cli.verify_approved(args, contract, ids, env)


def test_changed_original_is_rejected_after_review(prepared):
    args, contract, ids, env = prepared
    cli.prepare(args, contract, ids, env)
    args.candidate_sha256 = ledger.sha(args.candidate)
    approved = json.loads(args.candidate.read_bytes())
    approved.update(approved=True,
                    approval_id="r1b3-batch02-local-01-approved-v1",
                    reviewer_ref="independent-review-issue-xyz")
    approved["execution_intent"]["authorization_status"] = "APPROVED_FOR_BATCH02_LOCAL_QA"
    args.approval.write_bytes(ledger.encoded(approved))
    args.approval_sha256 = ledger.sha(args.approval)
    first = contract["source_files"][str(ids[0])]["rgb"]
    (args.cache / first).write_bytes(b"changed-original")
    with pytest.raises(ledger.Stop, match="source SHA pins"):
        cli.verify_approved(args, contract, ids, env)


def test_candidate_must_not_be_inside_protected_root(prepared):
    args, contract, ids, env = prepared
    args.output = args.cache / "candidate.json"
    with pytest.raises(ledger.Stop, match="overlaps protected"):
        cli.prepare(args, contract, ids, env)
    assert not args.output.exists()


def test_execute_requires_explicit_flag(monkeypatch):
    def forbidden(*unused):
        raise AssertionError("execution boundary was crossed")
    monkeypatch.setattr(cli, "context", forbidden)
    with pytest.raises(ledger.Stop, match="explicit --execute"):
        cli.main(["execute", "--batch", "2"])
    with pytest.raises(ledger.Stop, match="only for execute"):
        cli.main(["inspect", "--batch", "2", "--execute"])


def test_single_batch_dispatch_only_after_review(monkeypatch):
    batch_ids = list(range(50, 75))
    c = {"targets": {str(i): "train" for i in batch_ids}}
    args_env = {"git_dirty": False}
    monkeypatch.setattr(cli, "context",
                        lambda args: ({"status": "CLEAN_PAUSED"}, c, batch_ids, args_env))
    monkeypatch.setattr(cli, "verify_approved", lambda *unused: {"approved": True})
    received = []
    monkeypatch.setattr(cli.qa, "main",
                        lambda args, campaign_type=None: received.append((args, campaign_type)) or 2)
    rv = cli.main(["execute", "--batch", "2", "--execute",
                   "--approval", "approved.json", "--approval-sha256", "a" * 64,
                   "--candidate-sha256", "b" * 64])
    assert rv == 2
    cmd, campaign_type = received[0]
    assert cmd[0] == "run" and "--execute" in cmd
    assert cmd[cmd.index("--run-id") + 1] == "r1b3-batch02-local-01"
    assert cmd[cmd.index("--ids") + 1:] == [str(i) for i in batch_ids]
    assert campaign_type is cli.scope.ScopeCampaign
