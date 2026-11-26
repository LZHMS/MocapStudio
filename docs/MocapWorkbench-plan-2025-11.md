# MocapWorkbench 规划(fork 自 omp-studio @1e526b1)

上游参考克隆:`H:/refs/omp-studio`;本 fork:`H:/MocapWorkbench`。
数据层沿用 `H:/MocapStudio/server/app_server.py`(`/api/processing/*`,回环 8787)。

## 布局蓝图

- rail(左图标栏):看板 / 数据浏览器 / 队列 / FBX 查看 / 设置
- sidebar:数据浏览器卡(WorkspaceTree 接 `/api/processing/tree`,懒加载)+ 最近运行卡
- main:OMP dock 窗口系统原样(拖出浮窗、吸附分屏、最小化):
  - 看板窗(HomePage 改造:队列统计 tile、最近运行、数据源概览、管线阶段)
  - 队列窗(HistoryPage 改造:运行 master–detail)
  - 诊断窗(DiagnosticsPage 改造:阶段 tone 行 + meter + JSON 快照 + 日志)
  - FBX 查看窗(browser/iframe 承载 `/pipeline-viewer/fbx_viewer.html`)
- 顶栏命令输入:Composer 简化为 `@` 引用序列(`/api/processing/find` 模糊搜索)→ 加入待处理清单 → 建任务

## 功能裁决

- 保留原样:dock 窗口系统、shell/rail/tabs、WorkspaceTree+FileBrowser 视图层、styles/i18n/icons、CommandPalette、vitest
- 适配:StudioClient 数据层 → MocapClient(会话→jobs、workspace tree→/tree、DiagnosticsPage→阶段/报告、HistoryPage→队列、MentionMenu→序列搜索、Logs→worker 日志、HomePage→看板)
- 删除:Agent 对话全链路(Conversation/Composer 聊天/审批/BTW/语音/模型选择/AgentHub)、Git/Changes/GitHub、Time Travel、评测、Capabilities、ModelConfig、Media、SSH/WSL/手机、更新器、托管 runtime

## 工作流串联

Explorer 点选或 @序列 → 待处理清单 → 建任务(step/auto)→ 队列窗状态 → 诊断窗(tone 阶段行/meter/快照/日志)→ 产物下载 → FBX 查看窗

## 里程碑

- M0 fork 跑起来:`npm install` + `apps/renderer` vite dev + `?preview=1`(fixture 桥,无需 Rust runtime / submodule)
- M1 MocapClient 接真 API(tree/find/jobs/config),摘除对话主区
- M2 看板/队列/诊断/查看四窗布局 + rail 完成
- M3 一条序列 `G01-MP-S-WALK` 走通全 UI(mock worker 先行,真实 web_manager 随后)
- M4 删净死码、branding、(可选)Electron 打包

## 风险

- renderer 的 StudioClient 契约面大:用 `?preview=1` fixture 作基座,逐卡替换为真 API,不一次性删 App 大 switch(先摘导航与 dock 注册)
- `npm install` 依赖 Node 22+;本机 v24 ✓;vendor submodule 与 Rust host 仅在跑真实 runtime 时才需要(M4 之后再谈)
