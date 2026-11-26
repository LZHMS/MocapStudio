#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_export_overview.py -- Shogun 动捕全量数据状态大盘、目录结构优化与自动化回流工具。

功能：
1. 扫描控制目录对应的序列本地 processed/<version>/<take.mcp> 报告、修复批次及 Mesh 导出目录；
2. 修复回流机制：修复成功的序列自动提升为 SUCCESS，并从待复核清单中移入成功序列栏目；
3. 构建直观清晰的 Windows 快捷方式分类导航目录 (无损映射，不搬动原生大文件)；
4. 导出 Excel 可读的 CSV 清单及 Markdown 大盘统计报告；
5. 一键提取所有“检测成功且包含道具”的 HOI 序列清单，供 step6_attach_prop_meshes.py 挂载 Mesh。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from pipeline_layout import in_run, load_layout, report_paths, report_relative, source_runs, take_paths


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("H:/MocapData/_processing/processing"),
        help="批次控制目录 (默认: H:/MocapData/_processing/processing；也支持集中式通用输出)"
    )
    parser.add_argument(
        "--dashboard-dir",
        type=Path,
        default=None,
        help="生成的导航大盘根目录 (默认: <output-dir>/dashboard)"
    )
    parser.add_argument(
        "--mesh-export-dir",
        type=Path,
        default=Path("H:/MocapData/_processing"),
        help="Mesh 控制目录或包含多个 Mesh 批次的控制根目录 (默认: H:/MocapData/_processing)"
    )
    parser.add_argument(
        "--export-success-hoi-list",
        type=Path,
        default=None,
        help="可选输出路径：导出全部检测成功的 HOI 序列 JSON 清单 (供 step6_attach_prop_meshes 使用)"
    )
    parser.add_argument(
        "--no-shortcuts",
        action="store_true",
        help="不生成 Windows .lnk 快捷方式，仅刷新 CSV 和 Markdown 报表"
    )
    return parser.parse_args(argv)


def mesh_exports(mesh_dir: Path):
    """Yield exact take identities for local exports, legacy names otherwise."""
    if load_layout(mesh_dir) is None:
        controls = sorted(path for path in mesh_dir.iterdir()
                          if path.is_dir() and load_layout(path) is not None)
        if controls:
            for control in controls:
                if (control / "mesh_batch_report.json").is_file():
                    yield from mesh_exports(control)
            return
        for path in mesh_dir.glob("**/*_with_*.fbx"):
            yield None, path
        return
    if not (mesh_dir / "mesh_batch_report.json").is_file():
        return  # A normal processing control is not a mesh export batch.
    for rep in report_paths(mesh_dir):
        try:
            relative = report_relative(mesh_dir, rep)
            data = json.loads(rep.read_text(encoding="utf-8-sig"))
            if data.get("relative_path") is not None and Path(data["relative_path"]) != relative:
                continue
            paths = take_paths(mesh_dir, relative)
            roots = (paths["destination"].resolve(), paths["review_destination"].resolve())
            artifacts = data.get("artifacts", {})
            for role in sorted(artifacts, key=lambda role: (role not in ("scene_fbx", "fbx"), role)):
                value = artifacts[role]
                if not isinstance(value, str):
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = rep.parent / path
                if (path.suffix.casefold() == ".fbx" and in_run(mesh_dir, path)
                        and any(path.resolve().is_relative_to(root) for root in roots)
                        and path.is_file()):
                    yield relative.as_posix(), path
                    break  # Prefer the full-scene product over subject-only exports.
        except (OSError, TypeError, ValueError, AttributeError):
            continue


def scan_all_takes(output_dir: Path, mesh_dir: Path) -> tuple[dict[str, dict], dict]:
    output_dir = Path(output_dir)
    all_takes: dict[str, dict] = {}
    has_groups = any(d.is_dir() and d.name.startswith("group") for d in output_dir.iterdir()) if output_dir.is_dir() else False
    if has_groups:
        for rep in output_dir.glob("group*/**/report.json"):
            try:
                relative = rep.parent.relative_to(output_dir)
                rel = relative.as_posix()
                data = json.loads(rep.read_text(encoding="utf-8-sig"))
                status = data.get("status", "UNKNOWN")
                reasons = data.get("quality", {}).get("review_reasons", data.get("review_reasons", []))
                fbx_candidates = [f for f in rep.parent.glob("*.fbx") if not f.name.endswith("_with_mesh.fbx")]
                fbx_path = fbx_candidates[0] if fbx_candidates else None
                all_takes[rel] = {
                    "relative_path": rel,
                    "take_name": relative.name,
                    "source_status": status,
                    "final_status": status,
                    "repaired": bool(data.get("repaired") or status == "SUCCESS"),
                    "repair_run_name": None,
                    "report_path": str(rep.resolve()),
                    "artifacts_dir": str(rep.parent.resolve()),
                    "fbx_path": str(fbx_path.resolve()) if fbx_path else None,
                    "kind": "HOI" if "HOI" in relative.parts else "MP",
                    "group": relative.parts[0] if len(relative.parts) > 1 else "other",
                    "review_reasons": reasons,
                    "mesh_status": "NONE",
                    "mesh_fbx_path": None,
                }
            except (OSError, ValueError, TypeError, AttributeError):
                continue
    else:
        if load_layout(output_dir) is None and not (output_dir / "takes").is_dir():
            raise ValueError(f"Canonical report directory not found: {output_dir / 'takes'}")
        for run in source_runs(output_dir):
            repair = run != output_dir
            for rep in report_paths(run):
                try:
                    relative = report_relative(run, rep)
                    rel = relative.as_posix()
                    data = json.loads(rep.read_text(encoding="utf-8-sig"))
                    if data.get("relative_path") is not None and Path(data["relative_path"]) != relative:
                        raise ValueError("report identity disagrees with canonical path")
                    status = data.get("status", "UNKNOWN")
                    reasons = data.get("quality", {}).get("review_reasons", data.get("review_reasons", []))
                    art = data.get("artifacts", {})
                    fbx = art.get("scene_fbx") or art.get("fbx")
                    paths = take_paths(run, relative)
                    roots = (paths["destination"].resolve(), paths["review_destination"].resolve())
                    fbx_path = None
                    if fbx:
                        fbx_path = Path(fbx)
                        if not fbx_path.is_absolute():
                            fbx_path = rep.parent / fbx_path
                        fbx_path = fbx_path.resolve()
                        if not in_run(run, fbx_path) or not any(fbx_path.is_relative_to(root) for root in roots):
                            raise ValueError("FBX escapes this take's artifact roots")
                    artifact_dir = fbx_path.parent if fbx_path else (
                        paths["review_destination"] if status == "NEEDS_REVIEW" else paths["destination"])
                    if repair and rel in all_takes:
                        if status == "SUCCESS":
                            all_takes[rel].update(
                                final_status="SUCCESS", repaired=True, repair_run_name=run.name,
                                report_path=str(rep.resolve()), artifacts_dir=str(artifact_dir),
                                fbx_path=str(fbx_path) if fbx_path else None, review_reasons=[])
                        continue
                    all_takes[rel] = {
                        "relative_path": rel,
                        "take_name": relative.name,
                        "source_status": "REPAIR_ONLY" if repair else status,
                        "final_status": status,
                        "repaired": repair and status == "SUCCESS",
                        "repair_run_name": run.name if repair and status == "SUCCESS" else None,
                        "report_path": str(rep.resolve()),
                        "artifacts_dir": str(artifact_dir),
                        "fbx_path": str(fbx_path) if fbx_path else None,
                        "kind": "HOI" if "HOI" in relative.parts else "MP",
                        "group": relative.parts[0] if len(relative.parts) > 1 else "other",
                        "review_reasons": reasons,
                        "mesh_status": "NONE",
                        "mesh_fbx_path": None,
                    }
                except (OSError, ValueError, TypeError, AttributeError):
                    continue

    # 3. 关联 Mesh 挂载状态
    if mesh_dir.is_dir():
        for matched_rel, mesh_fbx in mesh_exports(mesh_dir):
            stem = mesh_fbx.stem
            if matched_rel is None:
                for rel in all_takes:
                    base = Path(rel).name.removesuffix(".mcp")
                    if base in stem:
                        matched_rel = rel
                        break
            if matched_rel in all_takes:
                tag = "WITH_MEETING_ROOM" if "with_meeting_room" in stem else "WITH_MESH"
                all_takes[matched_rel]["mesh_status"] = tag
                all_takes[matched_rel]["mesh_fbx_path"] = str(mesh_fbx.resolve())

    # 统计数据
    stats = {
        "total": len(all_takes),
        "success_total": len([t for t in all_takes.values() if t["final_status"] == "SUCCESS"]),
        "success_hoi": len([t for t in all_takes.values() if t["final_status"] == "SUCCESS" and t["kind"] == "HOI"]),
        "success_mp": len([t for t in all_takes.values() if t["final_status"] == "SUCCESS" and t["kind"] == "MP"]),
        "repaired_success": len([t for t in all_takes.values() if t["repaired"]]),
        "needs_review": len([t for t in all_takes.values() if t["final_status"] == "NEEDS_REVIEW"]),
        "failed": len([t for t in all_takes.values() if t["final_status"] == "FAILED"]),
        "with_mesh_count": len([t for t in all_takes.values() if t["mesh_status"] != "NONE"])
    }

    return all_takes, stats


def create_windows_shortcuts(entries: list[dict]):
    """通过 PowerShell COM 接口批量创建 Windows 原生快捷方式 (.lnk)"""
    script = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$entries = ConvertFrom-Json -InputObject ([Console]::In.ReadToEnd())
$shell = New-Object -ComObject WScript.Shell
foreach ($entry in $entries) {
    $dir = [System.IO.Path]::GetDirectoryName($entry.link_path)
    if (!(Test-Path $dir)) { [System.IO.Directory]::CreateDirectory($dir) | Out-Null }
    $link = $shell.CreateShortcut($entry.link_path)
    $link.TargetPath = $entry.target_path
    $link.Description = $entry.description
    $link.Save()
}
"""
    payload = json.dumps(entries, ensure_ascii=False)
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        input=payload,
        text=True,
        capture_output=True,
        check=False
    )


def generate_dashboard(output_dir: Path, dashboard_dir: Path, mesh_dir: Path, export_list: Path | None, create_shortcuts: bool):
    print("=" * 65)
    print(" 动捕全量数据状态扫描、目录优化与回流构建器")
    print("=" * 65)

    all_takes, stats = scan_all_takes(output_dir, mesh_dir)

    print(f"扫描完成：全量序列共 {stats['total']} 条")
    print(f"  ├─ [01 已成功序列 SUCCESS]     : {stats['success_total']} 条 (HOI: {stats['success_hoi']}, MP: {stats['success_mp']})")
    print(f"  │    └─ 其中经由修复回流成功   : {stats['repaired_success']} 条")
    print(f"  ├─ [02 待复核序列 NEEDS_REVIEW]: {stats['needs_review']} 条")
    print(f"  ├─ [03 失败序列 FAILED]       : {stats['failed']} 条")
    print(f"  └─ [04 已挂载 Mesh 最终交付产物]: {stats['with_mesh_count']} 条")

    dashboard_dir.mkdir(parents=True, exist_ok=True)

    # 1. 导出 CSV 清单 (Excel 友好)
    csv_path = dashboard_dir / "sequence_status_overview.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "序列标识 (relative_path)", "Take名称", "分组 (Group)", "类型 (HOI/MP)",
            "最终状态 (SUCCESS/REVIEW/FAILED)", "原始状态", "是否修复成功回流", "修复批次",
            "Mesh挂载状态", "审查/失败原因", "产物目录 (Artifacts Dir)", "主FBX路径", "带Mesh的FBX路径"
        ])
        for rel in sorted(all_takes.keys()):
            item = all_takes[rel]
            reasons = "; ".join(item["review_reasons"][:3]) if item["review_reasons"] else "None"
            writer.writerow([
                item["relative_path"], item["take_name"], item["group"], item["kind"],
                item["final_status"], item["source_status"], "是" if item["repaired"] else "否",
                item["repair_run_name"] or "-", item["mesh_status"], reasons,
                item["artifacts_dir"], item["fbx_path"] or "-", item["mesh_fbx_path"] or "-"
            ])
    print(f"\n[OK] CSV 全量大盘表已生成: {csv_path}")

    # 2. 导出 Markdown 看板
    md_path = dashboard_dir / "mocap_processing_dashboard.md"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# 动捕序列处理、修复与 Mesh 交付全景看板\n\n")
        f.write(f"> 统计基准时间：`{now_str}`\n\n")
        f.write(f"### 一、核心数据指标汇总\n\n")
        f.write(f"| 分类状态 | 总数 | HOI (双人交互) | MP (单人基元) | 说明 |\n")
        f.write(f"|---|---:|---:|---:|---|\n")
        f.write(f"| **全量序列** | **{stats['total']}** | - | - | 数据集入库总条数 |\n")
        f.write(f"| **01 已成功序列 (SUCCESS)** | **{stats['success_total']}** | **{stats['success_hoi']}** | **{stats['success_mp']}** | 完全通过质量门槛，包含修复回流 |\n")
        f.write(f"| └─ *其中经修复批次拯救成功* | *{stats['repaired_success']}* | *{len([t for t in all_takes.values() if t['repaired'] and t['kind']=='HOI'])}* | *{len([t for t in all_takes.values() if t['repaired'] and t['kind']=='MP'])}* | 已自动回流到成功序列分类中 |\n")
        f.write(f"| **02 待复核序列 (NEEDS_REVIEW)** | **{stats['needs_review']}** | - | - | 人体通过，存在刚体道具遮挡或丢点 |\n")
        f.write(f"| **03 失败序列 (FAILED)** | **{stats['failed']}** | - | - | 缺 X2D 文件或无头 Shogun 启动超时 |\n")
        f.write(f"| **04 已挂载 Mesh 最终交付序列** | **{stats['with_mesh_count']}** | - | - | 已注入动态道具与静态场景网格 |\n\n")
        f.write(f"### 二、目录结构导航指南\n\n")
        f.write(f"本目录已建立直观的 Windows 快捷方式导航，双击对应 `.lnk` 即可秒开底层原生目录：\n\n")
        f.write(f"- `01_已成功序列 (SUCCESS)/`: 全部通过验收的可用序列 (按 HOI / MP 细分)\n")
        f.write(f"- `02_修复成功回流序列 (REPAIRED_SUCCESS)/`: 从修复批次拯救成功的序列集中直达\n")
        f.write(f"- `03_待复核序列 (NEEDS_REVIEW)/`: 待进一步修复处理的序列\n")
        f.write(f"- `04_处理失败序列 (FAILED)/`: 失败序列诊断直达\n")
        f.write(f"- `05_已挂载Mesh交付物 (WITH_MESH)/`: 包含道具与会议室场景的最终 FBX\n")
    print(f"[OK] Markdown 看板已生成: {md_path}")

    # 3. 生成分类快捷方式目录
    if create_shortcuts:
        print("\n正在生成 Windows 分类快捷方式导航目录...")
        shortcut_entries = []

        for item in all_takes.values():
            take_name = item["take_name"]
            group = item["group"]
            kind = item["kind"]
            target = item["artifacts_dir"]

            if item["final_status"] == "SUCCESS":
                subfolder = "HOI_双人交互带道具" if kind == "HOI" else "MP_单人纯动作基元"
                link = dashboard_dir / "01_已成功序列 (SUCCESS)" / subfolder / group / f"{take_name}.lnk"
                shortcut_entries.append({
                    "link_path": str(link),
                    "target_path": target,
                    "description": f"SUCCESS Take: {item['relative_path']}"
                })
                # 如果是修复成功的，同时在修复回流目录做一份快速直达
                if item["repaired"]:
                    repair_link = dashboard_dir / "02_修复成功回流序列 (REPAIRED_SUCCESS)" / group / f"{take_name}.lnk"
                    shortcut_entries.append({
                        "link_path": str(repair_link),
                        "target_path": target,
                        "description": f"Repaired SUCCESS Take ({item['repair_run_name']}): {item['relative_path']}"
                    })

            elif item["final_status"] == "NEEDS_REVIEW":
                link = dashboard_dir / "03_待复核序列 (NEEDS_REVIEW)" / group / f"{take_name}.lnk"
                shortcut_entries.append({
                    "link_path": str(link),
                    "target_path": target,
                    "description": f"NEEDS_REVIEW Take: {item['relative_path']}"
                })

            elif item["final_status"] == "FAILED":
                link = dashboard_dir / "04_处理失败序列 (FAILED)" / group / f"{take_name}.lnk"
                shortcut_entries.append({
                    "link_path": str(link),
                    "target_path": target,
                    "description": f"FAILED Take: {item['relative_path']}"
                })

            # 如果已生成 Mesh，建立交付直达
            if item["mesh_fbx_path"]:
                mesh_link = dashboard_dir / "05_已挂载Mesh交付物 (WITH_MESH)" / group / f"{Path(item['mesh_fbx_path']).name}.lnk"
                shortcut_entries.append({
                    "link_path": str(mesh_link),
                    "target_path": str(Path(item["mesh_fbx_path"]).parent),
                    "description": f"Mesh FBX: {Path(item['mesh_fbx_path']).name}"
                })

        create_windows_shortcuts(shortcut_entries)
        print(f"[OK] 已构建 {len(shortcut_entries)} 个直达快捷方式，分类体系构建完成！")

    # 4. 导出“仅成功 HOI”序列清单供批量 Mesh 挂载管线直接读取
    if export_list:
        success_hoi_fbxs = []
        for item in sorted(all_takes.values(), key=lambda x: x["relative_path"]):
            if item["final_status"] == "SUCCESS" and item["kind"] == "HOI" and item["fbx_path"]:
                success_hoi_fbxs.append(item["fbx_path"])
        
        export_list.parent.mkdir(parents=True, exist_ok=True)
        export_list.write_text(json.dumps(success_hoi_fbxs, indent=2), encoding="utf-8")
        print(f"\n[OK] 已提取全部检测成功的 HOI 序列清单 ({len(success_hoi_fbxs)} 条): {export_list}")

    print("\n大盘与导航目录刷新全部就绪！")


if __name__ == "__main__":
    args = parse_arguments()
    dash = args.dashboard_dir or (args.output_dir / "dashboard")
    generate_dashboard(args.output_dir, dash, args.mesh_export_dir, args.export_success_hoi_list, not args.no_shortcuts)
