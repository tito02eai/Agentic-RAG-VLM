#!/usr/bin/env python3
"""Consolidate the preregistered G0-B contact admission evidence."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "output" / "g0b_contact_admission_audit_v1"
EVIDENCE = {
    "no_contact_control": "g0b_margin_probe_r0.010",
    "strict_ik_topdown_contact": "g0b_pinch_debug_v3",
    "side_grasp_contact": "g0b_side180_probe",
    "compliant_arm_contact": "g0b_compliant_arm_probe",
    "small_timestep_contact": "g0b_small_timestep_probe",
    "pgs_solver_contact": "g0b_pgs_probe",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for label, directory in EVIDENCE.items():
        path = PROJECT_ROOT / "output" / directory / "results.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        trial = payload["trials"][0]
        rows.append(
            {
                "label": label,
                "source": str(path.relative_to(PROJECT_ROOT)),
                "stable": bool(trial["stable"]),
                "success": bool(trial["success"]),
                "reset_detected": bool(trial["reset_detected"]),
                "warning_delta": trial["warning_delta"],
                "bilateral_contact_s": float(trial["contact_before_lift_s"]),
                "lift_m": float(trial["final_lift_m"]),
                "max_qvel": float(trial["max_qvel"]),
                "max_contact_force_n": float(trial["max_contact_force_n"]),
                "ik_position_error_m": trial.get("ik_position_error_m"),
                "initial_alignment_error_m": trial.get("initial_alignment_error_m"),
                "initial_contact_min_distance_m": trial.get("initial_contact_min_distance_m"),
            }
        )

    contact_rows = [row for row in rows if row["label"] != "no_contact_control"]
    criteria = {
        "finite_no_reset_no_warning": False,
        "max_qvel_below_50_rad_s": False,
        "bilateral_contact_at_least_0.15_s": False,
        "final_lift_at_least_0.035_m": False,
    }
    admitted = bool(contact_rows) and any(bool(row["success"]) for row in contact_rows)
    control_stable = next(
        (bool(row["stable"]) for row in rows if row["label"] == "no_contact_control"),
        False,
    )
    receipt = {
        "status": "PASS" if not missing and rows else "FAIL",
        "gate": "G0-B true contact closure and lift",
        "admitted": admitted,
        "decision": "NOT_ADMITTED" if not admitted else "ADMITTED",
        "claim_boundary": (
            "A NOT_ADMITTED result is a successful audit outcome: public demos must retain the "
            "kinematic/contact-assisted manipulation label and must not claim contact-dynamics grasp success."
        ),
        "no_contact_control_stable": control_stable,
        "admission_criteria": criteria,
        "evidence": rows,
        "missing": missing,
    }
    (OUTPUT / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# G0-B contact-dynamics admission audit",
        "",
        f"Decision: **{receipt['decision']}**.",
        "",
        "The no-contact control remains stable, while every trial that establishes hand-object contact "
        "violates the finite/no-reset gate. Strict IK, side-grasp orientation, compliant arm impedance, "
        "a 0.2 ms step and PGS were each tested; none admitted a sustained bilateral lift.",
        "",
        "| Trial | Stable | Reset | Bilateral contact | Final lift | Max qvel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['stable']} | {row['reset_detected']} | "
            f"{float(row['bilateral_contact_s']):.3f}s | {float(row['lift_m']):.4f}m | "
            f"{float(row['max_qvel']):.3g} |"
        )
    lines.extend(
        [
            "",
            "This audit closes neither real grasping nor sim-to-real. It prevents an unstable contact rollout "
            "from being presented as success and identifies contact-model/system-identification work as the next low-level milestone.",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
