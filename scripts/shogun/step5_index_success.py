"""Refresh a Windows Explorer success-only view without moving capture artifacts.

This is a snapshot, not a scheduler or quality recheck. Only managed shortcuts,
a CSV and ownership metadata under OutputDir/_success are written. Canonical
reports may live in sequence-local processed/<version>/<take.mcp> folders;
shortcuts target their actual artifacts, never a centralized compatibility tree.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from check_health import check_artifacts, read_json, validate_report
from step1_process_takes import write_json
from pipeline_layout import in_run, load_layout, report_paths, report_relative


OWNER = "shogun-success-view"
FIELDS = ("relative_path", "subjects", "artifacts_directory", "vdf", "fbx",
          "report_path", "shortcut_path")


def artifact_path(value, report_path):
    path = Path(value)
    return (path if path.is_absolute() else report_path.parent / path).resolve()


def collect_successes(output):
    """Use canonical reports and the same final-quality checks as check_health."""
    output = Path(output)
    if load_layout(output) is None and not (output / "takes").is_dir():
        raise ValueError(f"Canonical report directory is missing: {output / 'takes'}")
    entries, issues, seen, duplicates = [], [], set(), set()
    for report_path in report_paths(output):
        report, _, error = read_json(report_path)
        if error:
            issues.append(f"{report_path}: {error}")
            continue
        try:
            relative = report_relative(output, report_path)
            if report.get("relative_path") is not None and Path(report["relative_path"]) != relative:
                raise ValueError("report relative_path disagrees with canonical take identity")
        except (OSError, TypeError, ValueError) as exc:
            issues.append(f"{report_path}: {exc}")
            continue
        job_id = report.get("job_id")
        if isinstance(job_id, str):
            if job_id in seen:
                duplicates.add(job_id)
            seen.add(job_id)
        if report.get("status") != "SUCCESS":
            continue
        error = validate_report(report)
        missing = [] if error else check_artifacts(report, report_path, output)
        if error or missing:
            issues.append(f"{report_path}: {error or missing}")
            continue
        products = {role: artifact_path(value, report_path)
                    for role, value in report["artifacts"].items()}
        if any(not in_run(output, path) for path in products.values()):
            issues.append(f"{report_path}: SUCCESS artifact is outside run scope")
            continue
        shortcut = relative.with_name(relative.name + ".lnk")
        entries.append({
            "job_id": job_id, "relative_path": relative.as_posix(),
            "subjects": "; ".join(row["name"] for row in report.get("subjects", [])
                                  if row.get("processing_target") is True),
            "artifacts_directory": str(products["vdf"].parent),
            "vdf": str(products["vdf"]), "fbx": str(products["fbx"]),
            "report_path": str(report_path), "shortcut": shortcut.as_posix(),
            "shortcut_path": str(output / "_success" / shortcut),
        })
    if duplicates:
        entries = [entry for entry in entries if entry["job_id"] not in duplicates]
        issues.append(f"Ambiguous duplicate job IDs excluded: {sorted(duplicates)}")
    return sorted(entries, key=lambda entry: entry["relative_path"].casefold()), issues


def managed_link(view, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".lnk":
        raise ValueError(f"Invalid managed shortcut path: {relative}")
    target = view / path
    if not target.resolve().is_relative_to(view.resolve()):
        raise ValueError(f"Managed shortcut escapes the success view: {relative}")
    return target


def write_shortcuts(entries):
    """Create .lnk files via Windows Shell, never open their targets."""
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$entries = ConvertFrom-Json -InputObject ([Console]::In.ReadToEnd())
$shell = New-Object -ComObject WScript.Shell
foreach ($entry in $entries) {
    $link = $shell.CreateShortcut($entry.shortcut_path)
    $link.TargetPath = $entry.artifacts_directory
    $link.WorkingDirectory = $entry.artifacts_directory
    $link.Description = 'SUCCESS: ' + $entry.relative_path
    $link.Save()
}
"""
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        input=json.dumps(entries, ensure_ascii=False), capture_output=True, text=True,
        encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=120, check=True,
    )


def refresh(output, shortcut_writer=write_shortcuts):
    output = Path(output).expanduser().resolve()
    if not output.is_dir():
        raise ValueError(f"Output directory does not exist: {output}")
    view = output / "_success"
    if view.resolve() != view:
        raise ValueError("Success view must be an ordinary directory, not a link")
    manifest_path = view / "_index.json"
    previous = []
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("owner") != OWNER or metadata.get("output_dir") != str(output):
            raise ValueError("Existing success view belongs to another owner/output")
        previous = metadata["entries"]
    elif view.exists() and any(view.iterdir()):
        raise ValueError("Refusing to manage a nonempty _success directory without ownership metadata")
    entries, issues = collect_successes(output)
    old = {entry["shortcut"]: entry for entry in previous}
    desired = {entry["shortcut"]: entry for entry in entries}
    paths = {name: managed_link(view, name) for name in old.keys() | desired.keys()}
    for name in desired:
        if paths[name].exists() and name not in old:
            raise FileExistsError(f"Refusing to replace an unmanaged shortcut: {paths[name]}")
    metadata = {"owner": OWNER, "schema_version": 1, "output_dir": str(output),
                "updated_at": datetime.now(timezone.utc).isoformat(), "state": "refreshing",
                "entries": list((old | desired).values()), "issues": issues}
    # Record ownership before creating links so an interrupted refresh is recoverable.
    write_json(manifest_path, metadata)
    for name in desired:
        paths[name].parent.mkdir(parents=True, exist_ok=True)
    shortcut_writer(entries)
    for name in old.keys() - desired.keys():
        paths[name].unlink(missing_ok=True)  # .lnk is a file; never delete target data.
        parent = paths[name].parent
        while parent != view:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(entries)
    csv_path = view / "成功序列清单.csv"
    temporary = csv_path.with_name(csv_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(text.getvalue(), encoding="utf-8-sig", newline="")
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata.update(state="ready", entries=entries)
    write_json(manifest_path, metadata)
    return {"success_count": len(entries), "view": str(view), "csv": str(csv_path),
            "snapshot_at": metadata["updated_at"], "issues": issues,
            "note": "Snapshot only; run this command again to include newly finished sequences."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    options = parser.parse_args(argv)
    try:
        if os.name != "nt":
            raise ValueError("Windows is required to create Explorer shortcuts")
        result = refresh(options.output_dir)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Windows Shell: {exc.stderr.strip()}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
