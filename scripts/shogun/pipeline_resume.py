"""Read-only resume selection for Shogun pipeline take orchestration.

Identifies completed outputs with verified artifacts, safe retries, and
recoverable complete stages without modifying existing outputs or source data.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline_layout import take_paths


def _normalize_path(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve()).casefold()
    except Exception:
        return str(path).casefold()


def _is_safe_relative(path_str: str) -> bool:
    if not path_str or not isinstance(path_str, str):
        return False
    path = Path(path_str)
    return not path.is_absolute() and ".." not in path.parts


def select_resume_jobs(jobs: list[dict], output: Path) -> dict:
    """Classify jobs into preserved, pending, retry_reports, and recoverable stages.

    All inspection is strictly read-only.
    """
    output = Path(output).resolve()

    preserved: list[dict] = []
    pending: list[dict] = []
    retry_reports: list[dict] = []
    recoverable: list[tuple[dict, Path]] = []

    for job in jobs:
        paths = take_paths(output, job["relative_path"], data_root=job.get("layout_data_root"))
        work_root = paths["work_root"].resolve()
        roots = [paths[field].resolve() for field in ("destination", "review_destination")]
        expected_report = paths["report_path"].resolve()
        if Path(job["report_path"]).resolve() != expected_report:
            raise ValueError("Canonical report location differs from raw relative identity")
        # Preflight input errors (e.g. missing input file, missing X2D) stay attached.
        # They will be processed by run_jobs as failed records without launching native.
        if job.get("preflight_errors"):
            pending.append(job)
            continue

        report_path = Path(job["report_path"])
        if report_path.absolute() != report_path.resolve():
            raise ValueError("Canonical report must not traverse a symbolic link")
        job_id = job["job_id"]
        input_kind = job.get("input_kind", "mcp")
        required_artifacts = {"vdf", "fbx", "vsk"} if input_kind == "mcp" else {"vdf", "fbx"}

        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"Corrupt canonical report JSON: {report_path} ({exc})") from exc

            # Verify report identity matches this job.
            if report.get("job_id") != job_id:
                raise ValueError(
                    f"Job ID mismatch in canonical report {report_path}: "
                    f"expected {job_id}, found {report.get('job_id')}"
                )
            if _normalize_path(report.get("input_path")) != _normalize_path(job["input_path"]):
                raise ValueError(
                    f"Input path mismatch in canonical report {report_path}: "
                    f"expected {job['input_path']}, found {report.get('input_path')}"
                )

            status = report.get("status")
            if status in ("SUCCESS", "NEEDS_REVIEW"):
                products = report.get("artifacts")
                if not isinstance(products, dict) or not required_artifacts <= products.keys():
                    raise ValueError(
                        f"Canonical report missing required artifact keys: {report_path} "
                        f"(required {sorted(required_artifacts)})"
                    )

                # Every declared product must be a non-empty regular file.
                for role, art_path_str in products.items():
                    if not isinstance(art_path_str, str):
                        raise ValueError(f"Artifact {role} path is not string in {report_path}")
                    art_file = Path(art_path_str).resolve()
                    if not any(art_file.is_relative_to(root) for root in roots):
                        raise ValueError(f"Canonical artifact belongs to a different take: {art_file}")
                    if not art_file.is_file() or art_file.stat().st_size == 0:
                        raise ValueError(
                            f"Canonical report references missing or empty artifact: {art_file} "
                            f"(role {role} in {report_path})"
                        )

                # Check subject exports if present.
                for entry in report.get("subject_exports", []):
                    sub_path_str = entry.get("path")
                    if not isinstance(sub_path_str, str):
                        raise ValueError(f"Subject export path is not string in {report_path}")
                    sub_file = Path(sub_path_str).resolve()
                    if not any(sub_file.is_relative_to(root) for root in roots):
                        raise ValueError(f"Subject export belongs to a different take: {sub_file}")
                    if not sub_file.is_file() or sub_file.stat().st_size == 0:
                        raise ValueError(
                            f"Canonical report references missing or empty subject export: {sub_file} "
                            f"(in {report_path})"
                        )

                preserved.append(job)
                continue

            elif status == "FAILED":
                retry_reports.append({
                    "job_id": job_id,
                    "report_path": str(report_path),
                    "report_data": report,
                })
                pending.append(job)
                continue
            else:
                raise ValueError(f"Invalid or unrecognized status {status!r} in {report_path}")

        # Canonical report does not exist.
        # Check if there is a fully staged complete result in .work eligible for recovery.
        candidate_stages: list[Path] = []
        if work_root.is_dir():
            for stage in work_root.glob(f"{job_id}-*"):
                if not stage.is_dir() or not stage.resolve().is_relative_to(work_root):
                    continue
                result_file = stage / ".control" / "result.json"
                if not result_file.is_file() or not result_file.resolve().is_relative_to(stage.resolve()):
                    continue
                try:
                    res_data = json.loads(result_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if (res_data.get("job_id") == job_id
                        and _normalize_path(res_data.get("input_path")) == _normalize_path(job["input_path"])
                        and res_data.get("status") in ("SUCCESS", "NEEDS_REVIEW")):
                    prods = res_data.get("artifacts", {})
                    if isinstance(prods, dict) and required_artifacts <= prods.keys():
                        all_valid = True
                        for role in required_artifacts:
                            rel_str = prods.get(role)
                            if not _is_safe_relative(rel_str):
                                all_valid = False
                                break
                            target_file = stage / rel_str
                            if (not target_file.resolve().is_relative_to(stage.resolve())
                                    or not target_file.is_file() or target_file.stat().st_size == 0):
                                all_valid = False
                                break
                        if all_valid:
                            candidate_stages.append(stage)

        if len(candidate_stages) > 1:
            raise ValueError(
                f"Multiple competing complete stages found in .work for job {job_id}: "
                f"{[str(s) for s in candidate_stages]}"
            )
        if len(candidate_stages) == 1:
            recoverable.append((job, candidate_stages[0]))
            continue

        # No report and no recoverable stage.
        # Verify no orphan unrecorded artifacts exist in the destinations.
        for dest_key in ("destination", "review_destination"):
            dest_dir = Path(job[dest_key])
            if dest_dir.is_dir():
                orphans = [p for p in dest_dir.rglob("*") if p.is_file() and p.stat().st_size > 0]
                if orphans:
                    raise ValueError(
                        f"Unreported non-empty output files exist in {dest_dir} without canonical report: "
                        f"{orphans[:3]}"
                    )

        pending.append(job)

    return {
        "pending": pending,
        "preserved": preserved,
        "retry_reports": retry_reports,
        "recoverable": recoverable,
    }
