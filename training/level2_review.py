"""Level 2 approval gate. Never edits trainer or triggers training.

Create:  python training/level2_review.py propose
Approve: python training/level2_review.py approve --id <proposal-id> --confirm APPROVE
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from level2_checkpoint import validate_changes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONITOR = ROOT / "outputs_training_v5_level2" / "monitor"
DEFAULT_APPROVALS = ROOT / "outputs_training_v5_level2" / "approvals"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def propose_from_status(status):
    alerts = {a["code"] for a in status.get("alerts", [])}
    current = status.get("latest_validation")
    if not isinstance(current, dict):
        return None
    lr = current.get("learning_rate")
    if not isinstance(lr, (float, int)) or lr <= 0:
        return None

    if "early_stop_patience_exceeded" in alerts:
        # Do not blindly reduce an already tiny LR. A resumed, explicitly
        # approved stage resets early-stop count in train.py. This exploratory
        # option retains existing LR and only relaxes scheduler patience.
        changes = validate_changes({"scheduler_patience": 4})
        reason = (
            "已達 Early Stopping 門檻，但累計 Epoch 尚未到 50。"
            "可考慮核准一個探索性續訓階段：保留目前學習率與最佳權重，"
            "將 Scheduler Patience 設為 4 並重設早停計數。"
            "此變更不保證 mIoU 改善；請先檢查最佳權重。"
        )
        evidence = "early_stop_patience_exceeded"
    elif "possible_overfitting" in alerts:
        new_lr = max(1e-7, lr * 0.5)
        if abs(new_lr - lr) < 1e-12:
            return None
        changes = validate_changes({"learning_rate": new_lr})
        reason = "Train Loss 下降但 Val Loss 上升，下一個實驗嘗試減半 LR。"
        evidence = "possible_overfitting"
    elif "miou_plateau" in alerts:
        new_lr = max(1e-7, lr * 0.5)
        if abs(new_lr - lr) < 1e-12:
            return None
        changes = validate_changes({"learning_rate": new_lr})
        reason = "植被 mIoU 停滯；下一階段僅調整 LR。"
        evidence = "miou_plateau"
    else:
        # Never invent hyperparameter changes merely for low class IoU / NaN.
        return None
    payload = {
        "schema": "plant2co2-level2-proposal-v1",
        "status": "pending",
        "base_epoch": current.get("epoch"),
        "base_learning_rate": lr,
        "changes": changes,
        "reason": reason,
        "evidence": evidence,
        "source": "rule_based_monitor",
        "requires_human_approval": True,
        "scope": "next_launch_or_full_resume_only",
    }
    payload["id"] = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:16]
    return payload


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    temp.replace(path)


def propose(status_path, output):
    status = json.loads(status_path.read_text(encoding="utf-8"))
    payload = propose_from_status(status)
    if not payload:
        print("目前沒有符合既定規則的安全超參數方案；請保留現有設定並人工檢查報告。")
        return
    target = output / ("proposal_" + payload["id"] + ".json")
    write_json(target, payload)
    print(f"待審核方案：{target}")
    print(f"理由：{payload['reason']}")
    print(f"參數：{payload['changes']}")
    print(f"核准前不會套用。請執行 approve --id {payload['id']} --confirm APPROVE")


def approve(proposal_id, confirmation, output):
    if confirmation != "APPROVE":
        raise ValueError("需要完整輸入 --confirm APPROVE")
    if not proposal_id or any(c not in "0123456789abcdef" for c in proposal_id) or len(proposal_id) != 16:
        raise ValueError("無效的方案 ID")
    proposal_path = output / ("proposal_" + proposal_id + ".json")
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    if proposal.get("status") != "pending" or proposal.get("id") != proposal_id:
        raise ValueError("方案 ID 或狀態不正確")
    original_id = proposal.pop("id")
    if hashlib.sha256(canonical(proposal).encode("utf-8")).hexdigest()[:16] != original_id:
        raise ValueError("方案內容被修改，拒絕核准")
    changes = validate_changes(proposal.get("changes"))
    approved = {
        "status": "approved",
        "approval": "human-confirmed",
        "proposal_id": original_id,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "changes": changes,
    }
    target = output / ("approved_" + proposal_id + ".json")
    if target.exists():
        raise FileExistsError("此方案已核准，避免重複建立")
    write_json(target, approved)
    print(f"已核准但尚未套用：{target}")
    print("僅在手動啟動 train.py --approved-config <此檔案> 時才生效。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--status", type=Path, default=DEFAULT_MONITOR / "status.json")
    p.add_argument("--output", type=Path, default=DEFAULT_APPROVALS)
    a = sub.add_parser("approve")
    a.add_argument("--id", required=True)
    a.add_argument("--confirm", required=True)
    a.add_argument("--output", type=Path, default=DEFAULT_APPROVALS)
    args = parser.parse_args()
    if args.action == "propose":
        propose(args.status, args.output)
    else:
        approve(args.id, args.confirm, args.output)


if __name__ == "__main__":
    main()
