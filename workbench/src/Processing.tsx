import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';
import {
    processingApi, ProcessingApiError,
    type FindMatch, type FindResult, type InputKind, type InspectItem, type InspectPayload,
    type JobAction, type JobState, type OutputEntry, type ProcessingArtifact,
    type ProcessingConfig, type ProcessingJob, type ProcessingOptions, type ProcessingStage,
    type TreeResult,
} from './processingApi';
import AdvancedOptions from './AdvancedOptions';
import HomePanel from './HomePanel';
import { attachColumnResizers } from './resizer';
import './processing.css';

const TERMINAL: Partial<Record<JobState, true>> = { SUCCESS: true, NEEDS_REVIEW: true, FAILED: true, CANCELLED: true, INTERRUPTED: true };
const LARGE_FILE = 128 * 1024 * 1024;
const LOG_LIMIT = 512 * 1024;
const STATE_LABELS: Record<JobState, string> = {
    QUEUED: '排队中', STARTING: '启动中', WAITING: '等待操作', RUNNING: '执行中',
    PAUSE_REQUESTED: '等待安全暂停', SUCCESS: '完成', NEEDS_REVIEW: '需复核',
    FAILED: '失败', CANCELLED: '已取消', INTERRUPTED: '已中断',
};
// 简短标签 + 悬停完整说明：界面文字尽量短，语义不丢失。
const OPTION_LABELS = [
    { value: 'filter_markers', label: '标记点过滤', detail: '过滤标记点数据（适用于 MCP / VDF）' },
    { value: 'dataset_mode', label: '数据集模式', detail: '数据集模式 / 主题与预期名单审计（适用于 MCP、VDF）' },
    { value: 'attach_meshes', label: '挂载物体网格', detail: '挂载现有物体网格（需要 Blender 和对应资产）' },
    { value: 'lightweight', label: '轻量显示 FBX', detail: '生成轻量显示 FBX（无损显示裁剪；不修复源文件物理质量）' },
    { value: 'drop_unused_labeling', label: '移除未用 Labeling', detail: '轻量化时移除未使用的 Labeling；保留必需锚点' },
];
// 预期名单不再手填:0001-parse 与报告自动记录实际 subjects/props;MP/BODY 人体专用族删除全部物体。
const INITIAL_OPTIONS = ['filter_markers', 'dataset_mode', 'lightweight', 'drop_unused_labeling'];
const SOURCE_LABELS: Record<string, string> = { run_report: '处理报告', fbx_parse: 'FBX 结构', mcp_scan: 'MCP 扫描' };

type SelectedInput = { name: string; kind: InputKind };

function errorText(error: unknown): string {
    if (error instanceof ProcessingApiError && error.status === 409) {
        return `状态已变化或该操作当前不可用：${error.message}。已请求刷新，请按最新状态操作。`;
    }
    return error instanceof Error ? error.message : String(error);
}

function sizeText(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function timeText(value?: string): string {
    if (!value) return '尚未记录';
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}

function shortTime(value?: string): string {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.valueOf())) return value;
    return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}

// 挂钟时间(与章鱼时码器同钟):序列详情拍摄窗口用绝对时间对齐相机素材。
function wallClock(iso?: string): string {
    if (!iso) return '';
    const date = new Date(iso);
    if (Number.isNaN(date.valueOf())) return iso;
    return date.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
        + '.' + String(date.getMilliseconds()).padStart(3, '0');
}

function statusClass(state: string): string {
    if (state === 'SUCCESS') return 'green';
    if (state === 'FAILED' || state === 'INTERRUPTED') return 'red';
    if (state === 'NEEDS_REVIEW' || state === 'PAUSE_REQUESTED') return 'amber';
    if (state === 'RUNNING' || state === 'STARTING') return 'blue';
    return 'faint';
}

function artifactRoleLabel(artifact: ProcessingArtifact): string {
    switch (artifact.role) {
        case 'archive_export_fbx': case 'export_fbx': return '03_exports · 直接可用';
        case 'archive_processed_fbx': return '02_processed · 完整工作档';
        case 'display_fbx': return '显示裁剪(旧任务遗留)';
        case 'fbx': return '完整导出(任务目录)';
        default: return artifact.role.startsWith('subject_') ? '逐人序列' : artifact.role;
    }
}

function viewerPreferred(artifacts: ProcessingArtifact[]): ProcessingArtifact | null {
    const fbxs = artifacts.filter(isFbx);
    if (!fbxs.length) return null;
    // 03_exports 的可用版优先(lightweight 直写 export_fbx,或归档回退 archive_export_fbx),
    // 绝不默认打开 processed/任务目录里的完整工作版。
    return fbxs.find((entry) => entry.role === 'export_fbx')
        ?? fbxs.find((entry) => entry.role === 'archive_export_fbx')
        ?? fbxs.find((entry) => entry.role === 'display_fbx')
        ?? fbxs.find((entry) => entry.role.startsWith('subject_'))
        ?? fbxs.find((entry) => entry.role === 'fbx')
        ?? fbxs[0];
}

function isFbx(artifact: ProcessingArtifact): boolean {
    return /\.fbx$/i.test(artifact.name) || /\.fbx$/i.test(artifact.path);
}

function stageSnapshot(report: Record<string, unknown> | null, stage: ProcessingStage, stages: ProcessingStage[]): unknown {
    if (!Array.isArray(report?.stages)) return null;
    const entries = report.stages.filter((entry): entry is Record<string, unknown> => !!entry && typeof entry === 'object');
    const exact = entries.find((entry) => entry.id === stage.id);
    if (exact) return exact;
    const occurrence = stages.slice(0, stages.findIndex((entry) => entry.id === stage.id))
        .filter((entry) => entry.name === stage.name && entry.subject === stage.subject).length;
    return entries.filter((entry) => entry.name === stage.name && (entry.subject ?? undefined) === stage.subject)[occurrence] ?? null;
}

function gbytesT(bytes: number): string {
    if (bytes >= 1024 ** 4) return `${(bytes / 1024 ** 4).toFixed(2)} TB`;
    return `${(bytes / 1024 ** 3).toFixed(0)} GB`;
}

type StageGroupItem = { stage: ProcessingStage; index: number };
type StageGroup = { kind: 'single'; stage: ProcessingStage; index: number } | { kind: 'group'; key: string; label: string; items: StageGroupItem[] };

function buildStageGroups(stages: ProcessingStage[]): StageGroup[] {
    const out: StageGroup[] = [];
    let i = 0;
    while (i < stages.length) {
        const stage = stages[i];
        if (stage.subject) {
            let j = i + 1;
            while (j < stages.length && stages[j].name === stage.name && stages[j].subject) j += 1;
            if (j - i >= 2) {
                out.push({ kind: 'group', key: `g:${stage.name}:${stage.id}`, label: stage.label || stage.name, items: stages.slice(i, j).map((item, offset) => ({ stage: item, index: i + offset })) });
                i = j;
                continue;
            }
        }
        out.push({ kind: 'single', stage, index: i });
        i += 1;
    }
    return out;
}

function groupSummary(items: StageGroupItem[]): { tone: string; text: string; seconds: number } {
    let seconds = 0;
    for (const item of items) seconds += item.stage.elapsed_seconds ?? 0;
    if (items.some((item) => item.stage.status === 'FAILED')) return { tone: 'red', text: '有失败', seconds };
    if (items.some((item) => item.stage.status === 'RUNNING')) return { tone: 'blue', text: '执行中', seconds };
    if (items.every((item) => item.stage.status === 'SKIPPED')) return { tone: 'faint', text: '全部跳过', seconds };
    if (items.every((item) => item.stage.status === 'WAITING')) return { tone: 'faint', text: '等待', seconds };
    return { tone: 'green', text: '完成', seconds };
}

interface ReportRow { name?: string; rig_class?: string; classification?: string; review_reasons?: unknown; quality?: { usable?: boolean } }

function issueSeverity(what: string, why: string): 'crit' | 'warn' | 'notice' {
    if (what.startsWith('阶段') || /error|exception|checkpoint|LOST|ZERO_DATA|unavailable/i.test(why)) return 'crit';
    if (/NOT_APPLICABLE|not_configured|no_retargeting|disabled/i.test(why) && !/ZERO_DATA|LOST/.test(why)) return 'notice';
    return 'warn';
}

function ReportSummary({ report }: { report: Record<string, unknown> | null }) {
    if (!report) return null;
    const subjects = Array.isArray(report.subjects) ? report.subjects as ReportRow[] : [];
    const props = Array.isArray(report.props) ? report.props as ReportRow[] : [];
    const topReasons = Array.isArray(report.review_reasons) ? (report.review_reasons as unknown[]).map(String) : [];
    const quality = report.quality as { usable?: boolean } | undefined;
    const status = String(report.status ?? '');
    const issues: { what: string; why: string; sev: 'crit' | 'warn' | 'notice' }[] = [];
    const addIssue = (what: string, why: string) => issues.push({ what, why, sev: issueSeverity(what, why) });
    for (const row of [...subjects, ...props]) {
        const reasons = Array.isArray(row.review_reasons) ? (row.review_reasons as unknown[]).map(String) : [];
        const unusable = row.quality && row.quality.usable === false;
        if (!unusable && !reasons.length) continue;
        const why = [...reasons, unusable && !reasons.length ? 'quality.usable=false' : ''].filter(Boolean).join('; ');
        addIssue(`${row.name ?? '?'}(${row.rig_class === 'HUMAN_RIG' ? '人物' : '物体'})`, why + (row.classification ? ` · ${row.classification}` : ''));
    }
    const stages = Array.isArray(report.stages) ? report.stages as { name?: string; label?: string; subject?: string; status?: string; error?: string; checkpoint_error?: string }[] : [];
    for (const stage of stages) {
        if (stage.status === 'FAILED' || stage.error || stage.checkpoint_error) {
            addIssue(`阶段 ${stage.label || stage.name}${stage.subject ? ` · ${stage.subject}` : ''}`, stage.error || stage.checkpoint_error || '失败');
        }
    }
    const sevCount = { crit: 0, warn: 0, notice: 0 };
    for (const issue of issues) sevCount[issue.sev] += 1;
    const total = issues.length + topReasons.length;
    const running = ['RUNNING', 'STARTING', 'QUEUED'].includes(status) || Boolean(report.pipeline_status_pending);
    const stateTone = running ? 'blue' : status === 'SUCCESS' ? 'green' : status === 'NEEDS_REVIEW' ? 'amber' : status === 'FAILED' ? 'red' : 'faint';
    const totalTone = sevCount.crit > 0 ? 'red' : sevCount.warn > 0 ? 'amber' : sevCount.notice > 0 ? 'blue' : 'green';
    return <section className="report-card" aria-label="处理报告">
        <div className="report-head">
            <h3>处理报告</h3>
            <span className={`chip ${stateTone}`}>{running ? '处理中' : status === 'SUCCESS' ? '通过' : status === 'NEEDS_REVIEW' ? '需复核' : status === 'FAILED' ? '失败' : status || '进行中'}</span>
            <span className={`chip ${running ? 'faint' : quality ? (quality.usable ? 'green' : 'red') : 'faint'}`}>质量 {running ? '待完成评估' : quality ? (quality.usable ? '可用' : '不可用') : '未评估'}</span>
            {!running && total > 0 && <span className={`chip ${totalTone}`}>{total} 个问题</span>}
            {!running && sevCount.crit > 0 && <span className="chip red">严重 {sevCount.crit}</span>}
            {!running && sevCount.warn > 0 && <span className="chip amber">警告 {sevCount.warn}</span>}
            {!running && sevCount.notice > 0 && <span className="chip blue">提示 {sevCount.notice}</span>}
        </div>
        {running
            ? <p className="report-pending">任务正在运行:阶段结果持续写入,完成后再给质量结论与复核清单。</p>
            : total === 0
            ? <p className="report-good">所有阶段、人物与道具均通过检查,没有需要人工复核的项。</p>
            : <>
                {!!topReasons.length && <div className="report-reasons">{topReasons.map((reason, index) => <span key={`t${index}`} className={`report-reason global sev-${issueSeverity('全局', reason)}`}>{reason}</span>)}</div>}
                {!!issues.length && <ul className="report-issues">{issues.slice(0, 48).map((issue, index) => <li key={index} className={`sev-${issue.sev}`}><strong>{issue.what}</strong><span>{issue.why}</span></li>)}</ul>}
                {issues.length > 48 && <p className="hint">仅显示前 48 条；完整清单见下方报告 JSON。</p>}
            </>}
    </section>;
}

function stageStatusText(status: string): string {
    return status === 'SUCCESS' ? '完成' : status === 'FAILED' ? '失败' : status === 'SKIPPED' ? '跳过'
        : status === 'RUNNING' ? '执行中' : '等待';
}

function ModuleCard({ title, count, open, onToggle, tone, children }: { title: string; count?: number; open?: boolean; onToggle?: (next: boolean) => void; tone?: string; children: ReactNode }) {
    const [inner, setInner] = useState(false);
    const isOpen = open === undefined ? inner : open;
    return <section className="module-card">
        <button type="button" className="module-head" aria-expanded={isOpen}
            onClick={() => (onToggle ? onToggle(!isOpen) : setInner((previous) => !previous))}>
            <h3>{title}</h3>
            {count != null && <span className={`chip ${tone ?? ''}`}>{count}</span>}
            <span className="grow" />
            <span className={`tw${isOpen ? ' open' : ''}`} aria-hidden="true" />
        </button>
        {isOpen && <div className="module-body">{children}</div>}
    </section>;
}

function ArtifactLink({ artifact }: { artifact: ProcessingArtifact }) {
    return <a className="artifact-link" href={artifact.url} download={artifact.name} title={`下载 ${artifact.path}`}>
        <i className="ico ico-down" aria-hidden="true" />
        <span>{artifact.name}<small>{artifactRoleLabel(artifact)} · {sizeText(artifact.size_bytes)}</small></span>
    </a>;
}

function WorkersSelect({ value, max, disabled, onChange }: { value: number; max: number; disabled: boolean; onChange: (value: string) => void }) {
    const [open, setOpen] = useState(false);
    const root = useRef<HTMLSpanElement | null>(null);
    useEffect(() => {
        if (!open) return;
        const away = (event: MouseEvent) => { if (root.current && !root.current.contains(event.target as Node)) setOpen(false); };
        const key = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false); };
        window.addEventListener('mousedown', away);
        window.addEventListener('keydown', key);
        return () => { window.removeEventListener('mousedown', away); window.removeEventListener('keydown', key); };
    }, [open]);
    const options = Array.from({ length: Math.max(1, Math.min(8, max)) }, (_, index) => index + 1);
    return <span className="num-select" ref={root}>
        <button type="button" className="btn outline num-select-btn" disabled={disabled}
            aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((previous) => !previous)}>
            {value}<span className="num-caret" aria-hidden="true" />
        </button>
        {open && <span className="num-select-panel" role="listbox">
            {options.map((option) => <button key={option} type="button" role="option" aria-selected={option === value}
                className={option === value ? 'active' : ''}
                onClick={() => { onChange(String(option)); setOpen(false); }}>{option}</button>)}
        </span>}
    </span>;
}

const JSON_SCALAR_CAP = 400;
const JSON_DEPTH = 8;

function JsonValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
    if (value === null) return <span className="j-null">null</span>;
    if (typeof value !== 'object') {
        return <span className={typeof value === 'string' ? 'j-str' : 'j-num'}>{typeof value === 'string' ? JSON.stringify(value) : String(value)}</span>;
    }
    if (depth >= JSON_DEPTH) return <span className="j-punct">{Array.isArray(value) ? `[…${value.length} 项]` : '{…}'}</span>;
    if (Array.isArray(value)) {
        if (!value.length) return <span className="j-punct">[]</span>;
        const allScalar = value.every((item) => item === null || typeof item !== 'object');
        if (allScalar) {
            const shown = value.length > JSON_SCALAR_CAP ? value.slice(0, JSON_SCALAR_CAP) : value;
            return <span className="j-inline">[{shown.map((item, index) => <span key={index}><JsonValue value={item} depth={depth + 1} />{index < shown.length - 1 ? ', ' : ''}</span>)}
                {value.length > shown.length ? <span className="j-more">, …共 {value.length} 项</span> : ''}]</span>;
        }
        return <span className="j-punct">[<span className="j-block">{value.map((item, index) => (
            <span key={index} className="j-line"><JsonValue value={item} depth={depth + 1} />{index < value.length - 1 ? ',' : ''}</span>))}</span>]</span>;
    }
    const entries = Object.entries(value as Record<string, unknown>);
    if (!entries.length) return <span className="j-punct">{'{}'}</span>;
    const allScalar = entries.every(([, item]) => item === null || typeof item !== 'object');
    if (allScalar && entries.length >= 6) {
        const cap = 120;
        const shown = entries.length > cap ? entries.slice(0, cap) : entries;
        return <span className="j-punct">{'{'}<span className="j-inline">{shown.map(([key, item], index) => (
            <span key={key}><span className="j-key">{JSON.stringify(key)}</span>: <JsonValue value={item} depth={depth + 1} />{index < shown.length - 1 ? ', ' : ''}</span>))}
            {entries.length > shown.length ? <span className="j-more">, …共 {entries.length} 键</span> : ''}</span>{'}'}</span>;
    }
    return <span className="j-punct">{'{'}<span className="j-block">{entries.map(([key, item], index) => (
        <span key={key} className="j-line"><span className="j-key">{JSON.stringify(key)}</span>: <JsonValue value={item} depth={depth + 1} />{index < entries.length - 1 ? ',' : ''}</span>))}</span>{'}'}</span>;
}

function JsonDiagnostic({ value }: { value: unknown }) {
    return <pre className="json-view"><JsonValue value={value} /></pre>;
}

export default function Processing() {
    const [view, setView] = useState<'home' | 'work' | 'viewer'>('home');
    // 浏览器返回支持:视图切换写入 history,返回键回到上一视图而非退出站点。
    const navDepth = useRef(0);
    const skipPush = useRef(false);
    useEffect(() => {
        history.replaceState({ view: 'home' }, '');
        const onPop = (event: PopStateEvent) => {
            const state = event.state as { view?: 'home' | 'work' | 'viewer' } | null;
            setView(state && state.view ? state.view : 'home');
            navDepth.current = Math.max(0, navDepth.current - 1);
            skipPush.current = true;
        };
        window.addEventListener('popstate', onPop);
        return () => window.removeEventListener('popstate', onPop);
    }, []);
    useEffect(() => {
        if (skipPush.current) { skipPush.current = false; return; }
        history.pushState({ view }, '');
        navDepth.current += 1;
    }, [view]);
    function navigateBack() {
        if (navDepth.current >= 1) history.back();
        else setView('home');
    }
    const colsRef = useRef<HTMLDivElement | null>(null);
    // 左列下部「处理区」高度:可拖分隔条调整并持久化,默认给足空间。
    const [trayHeight, setTrayHeight] = useState<number>(() => {
        const saved = Number(window.localStorage.getItem('mocap-tray-h'));
        return Number.isFinite(saved) && saved >= 140 && saved <= 1200 ? saved : 420;
    });
    const trayBounds = useRef({ min: 140, max: 1200 });
    function clampTray(value: number) {
        const { min, max } = trayBounds.current;
        return Math.min(max, Math.max(min, value));
    }
    function persistTray(value: number) { window.localStorage.setItem('mocap-tray-h', String(Math.round(value))); }
    function onTraySplitDown(event: ReactPointerEvent<HTMLDivElement>) {
        const splitter = event.currentTarget;
        const aside = splitter.parentElement;
        if (!aside) return;
        const asideHeight = aside.getBoundingClientRect().height;
        trayBounds.current = { min: 140, max: Math.max(160, Math.round(asideHeight - 260)) };
        const startY = event.clientY;
        const startH = trayHeight;
        try { splitter.setPointerCapture(event.pointerId); } catch { /* 合成事件下捕获可选 */ }
        splitter.classList.add('is-dragging');
        let latest = startH;
        const move = (moveEvent: PointerEvent) => {
            latest = clampTray(startH - (moveEvent.clientY - startY));
            setTrayHeight(latest);
        };
        const up = () => {
            splitter.classList.remove('is-dragging');
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', up);
            window.removeEventListener('pointercancel', up);
            persistTray(latest);
        };
        window.addEventListener('pointermove', move);
        window.addEventListener('pointerup', up);
        window.addEventListener('pointercancel', up);
    }
    const [config, setConfig] = useState<ProcessingConfig | null>(null);
    const [configError, setConfigError] = useState('');
    const [configEpoch, setConfigEpoch] = useState(0);
    const [workers, setWorkers] = useState('1');
    const [savingWorkers, setSavingWorkers] = useState(false);

    // 数据浏览器（懒加载资源管理器树）。
    const [rootPath, setRootPath] = useState('');
    const [treeData, setTreeData] = useState<Record<string, TreeResult>>({});
    const [treeBusy, setTreeBusy] = useState<Record<string, true>>({});
    const [treeErrors, setTreeErrors] = useState<Record<string, string>>({});
    const [expanded, setExpanded] = useState<Record<string, true>>({});
    const [searchText, setSearchText] = useState('');
    const treeControllers = useRef(new Map<string, AbortController>());
    const treeRefs = useRef(new Map<string, HTMLButtonElement>());

    // 待处理清单与 @ 跨根搜索（同一个输入框：@ 前缀=跨根模糊搜索）。
    const [selectedInputs, setSelectedInputs] = useState<Record<string, SelectedInput>>({});
    const [mention, setMention] = useState<FindResult | null>(null);
    const [mentionBusy, setMentionBusy] = useState(false);
    const [mentionIndex, setMentionIndex] = useState(0);
    const [mentionError, setMentionError] = useState('');
    const mentionTimer = useRef<number | undefined>(undefined);
    const mentionController = useRef<AbortController | null>(null);

    const [optionFlags, setOptionFlags] = useState<string[]>(INITIAL_OPTIONS);
    const [jitter, setJitter] = useState('0');
    const [tolerance, setTolerance] = useState('2');
    const [creating, setCreating] = useState(false);

    const [jobs, setJobs] = useState<ProcessingJob[]>([]);
    const [collapsed, setCollapsed] = useState<Record<string, boolean>>({ done: true });
    const [jobsLoading, setJobsLoading] = useState(true);
    const [jobsError, setJobsError] = useState('');
    const [jobsRefreshed, setJobsRefreshed] = useState('');
    const [selectedId, setSelectedId] = useState('');
    const [detail, setDetail] = useState<ProcessingJob | null>(null);
    const [approveOpen, setApproveOpen] = useState(false);
    const [approveNote, setApproveNote] = useState('');
    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState('');
    const [report, setReport] = useState<Record<string, unknown> | null>(null);
    const [reportError, setReportError] = useState('');
    const [reportOpen, setReportOpen] = useState(false);
    const [optionsOpen, setOptionsOpen] = useState(false);
    const [artifactsOpen, setArtifactsOpen] = useState(true);
    const [logsOpen, setLogsOpen] = useState(false);
    const [logs, setLogs] = useState('');
    const [logsTrimmed, setLogsTrimmed] = useState(false);
    const [logsError, setLogsError] = useState('');
    const [followLogs, setFollowLogs] = useState(true);
    const logsContainer = useRef<HTMLDivElement | null>(null);
    const [stageId, setStageId] = useState('');
    const [groupOpen, setGroupOpen] = useState<Record<string, boolean>>({});
    const [pendingAction, setPendingAction] = useState<JobAction | null>(null);
    const [actionError, setActionError] = useState('');
    const [notice, setNotice] = useState('');
    const [refreshEpoch, setRefreshEpoch] = useState(0);

    // FBX 查看覆盖层。
    const [viewerSource, setViewerSource] = useState('');
    const [viewerLoading, setViewerLoading] = useState(false);
    const [viewerError, setViewerError] = useState('');
    const [fbxFiles, setFbxFiles] = useState<OutputEntry[] | null>(null);
    const [chain, setChain] = useState<string[]>([]);
    const [currentPath, setCurrentPath] = useState('');

    // 序列详情覆盖层（subjects / objects 解析）。
    const [inspectPath, setInspectPath] = useState('');
    const [inspectData, setInspectData] = useState<InspectPayload | null>(null);
    const [inspectRange, setInspectRange] = useState<Record<string, number> | null>(null);
    const [inspectBusy, setInspectBusy] = useState(false);
    const [inspectError, setInspectError] = useState('');
    const [inspectItemKey, setInspectItemKey] = useState('');
    const inspectController = useRef<AbortController | null>(null);

    const openInspect = useCallback((path: string) => {
        inspectController.current?.abort();
        const controller = new AbortController();
        inspectController.current = controller;
        setInspectPath(path);
        setInspectData(null);
        setInspectRange(null);
        setInspectError('');
        setInspectItemKey('');
        setInspectBusy(true);
        processingApi.inspect(path, controller.signal).then((result) => {
            if (controller.signal.aborted) return;
            setInspectData(result);
            const first = result.subjects[0] ? ['s', result.subjects[0].name]
                : result.objects[0] ? ['o', result.objects[0].name] : null;
            if (first) setInspectItemKey(first.join('|'));
        }).catch((error: unknown) => {
            if (!controller.signal.aborted) setInspectError(errorText(error));
        }).finally(() => {
            if (!controller.signal.aborted) setInspectBusy(false);
        });
    }, []);

    const closeInspect = useCallback(() => {
        inspectController.current?.abort();
        setInspectPath('');
        setInspectData(null);
        setInspectError('');
        setInspectItemKey('');
        setInspectBusy(false);
    }, []);

    useEffect(() => {
        if (!inspectPath) return;
        function onKey(event: KeyboardEvent) {
            if (event.key === 'Escape') closeInspect();
        }
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [inspectPath, closeInspect]);

    // 序列有任务在执行时,覆盖层每 4 秒拉一次最新名单(报告随阶段更新)。
    const inspectActive = !!inspectData && inspectData.runs.some(run => !TERMINAL[run.state as JobState]);
    useEffect(() => {
        if (!inspectPath || !inspectActive) return;
        const timer = window.setInterval(() => {
            processingApi.inspect(inspectPath).then((result) => {
                if (result.path === inspectPath) {
                    setInspectData(result);
                    setInspectRangeTick((tick) => tick + 1);
                }
            }).catch(() => { /* 错误保留上一份数据。 */ });
        }, 4000);
        return () => window.clearInterval(timer);
    }, [inspectPath, inspectActive]);

    const refreshTree = () => {
        if (!rootPath) return;
        loadTree(rootPath);
        Object.keys(expanded).forEach((path) => { if (path !== rootPath) loadTree(path); });
    };
    async function deleteOutputFile(entry: { name: string; path: string }) {
        if (!config) return;
        if (!window.confirm(`删除处理产出 ${entry.name}?\n只删除结果目录(02_*/03_*)下的这份文件;任务目录与源数据不受影响。不可撤销。`)) return;
        try {
            await processingApi.deleteOutput(entry.path, config.csrf_token);
            setNotice(`已删除 ${entry.name}。`);
            setSelectedInputs((previous) => {
                if (!(entry.path in previous)) return previous;
                const next = { ...previous };
                delete next[entry.path];
                return next;
            });
            loadTree(entry.path.replace(/[\\/][^\\/]+$/, ''));
        } catch (error) { setActionError(errorText(error)); }
    }
    // 报告时间范围:报告里 range 在 load 阶段后写入,直接拉原始报告重点展示。
    const [inspectRangeTick, setInspectRangeTick] = useState(0);
    useEffect(() => {
        const jobId = inspectData?.report_job;
        if (!inspectPath || !jobId) return;
        let stopped = false;
        processingApi.report(jobId).then((result) => {
            if (stopped) return;
            const raw = result.report as Record<string, unknown> | null;
            const range = raw && typeof raw === 'object' ? raw.range : null;
            if (range && typeof range === 'object') setInspectRange(range as Record<string, number>);
        }).catch(() => { /* 报告尚未生成 */ });
        return () => { stopped = true; };
    }, [inspectPath, inspectData?.report_job, inspectRangeTick]);

    // 列拖拽调宽(仅处理台视图挂载;切回主页时自动 detach)。
    useEffect(() => {
        if (view !== 'work') return;
        const host = colsRef.current;
        if (!host) return;
        return attachColumnResizers(host);
    }, [view]);

    // 顶栏实时环境芯片:CPU/内存占用 + 数据盘可用空间(每 2.5s)。
    const [metrics, setMetrics] = useState<{ cpu_percent: number | null; memory_used_bytes: number | null; memory_total_bytes: number | null; disk: { path: string; total_bytes: number; free_bytes: number }[] } | null>(null);
    useEffect(() => {
        let stopped = false;
        async function poll() {
            try {
                const result = await processingApi.metrics();
                if (!stopped) setMetrics(result.metrics);
            } catch { /* 下一轮再试 */ }
        }
        void poll();
        const timer = window.setInterval(() => void poll(), 2500);
        return () => { stopped = true; window.clearInterval(timer); };
    }, []);

    const loadTree = useCallback((path: string | null = null) => {
        const key = path ?? 'root';
        treeControllers.current.get(key)?.abort();
        const controller = new AbortController();
        treeControllers.current.set(key, controller);
        setTreeBusy((previous) => ({ ...previous, [key]: true }));
        processingApi.tree(path, controller.signal).then((result) => {
            if (controller.signal.aborted) return;
            setTreeData((previous) => ({ ...previous, [result.path]: result }));
            setTreeErrors((previous) => {
                const next = { ...previous };
                delete next[path ?? 'root'];
                return next;
            });
            if (!path) setRootPath(result.path);
        }).catch((error: unknown) => {
            if (!controller.signal.aborted) setTreeErrors((previous) => ({ ...previous, [key]: errorText(error) }));
        }).finally(() => {
            if (!controller.signal.aborted) setTreeBusy((previous) => {
                const next = { ...previous };
                delete next[key];
                return next;
            });
        });
    }, []);

    useEffect(() => {
        const controller = new AbortController();
        setConfigError('');
        processingApi.config(controller.signal).then((result) => {
            if (controller.signal.aborted) return;
            setConfig(result);
            setWorkers(String(result.max_workers));
            setRootPath(result.default_browse_path);
            loadTree(result.default_browse_path);
        }).catch((error: unknown) => {
            if (!controller.signal.aborted) setConfigError(errorText(error));
        });
        return () => controller.abort();
    }, [configEpoch, loadTree]);

    useEffect(() => () => {
        treeControllers.current.forEach((controller) => controller.abort());
        mentionController.current?.abort();
    }, []);

    useEffect(() => {
        const controller = new AbortController();
        let timer: number | undefined;
        async function refresh() {
            try {
                const result = await processingApi.jobs(controller.signal);
                if (controller.signal.aborted) return;
                setJobs([...result.jobs].sort((a, b) => b.created_at.localeCompare(a.created_at)));
                // 队列不变式:每序列一个任务;选中任务被引擎去重移除时回退到第一条。
                setSelectedId((current) => (current && result.jobs.some((entry) => entry.id === current))
                    ? current : (result.jobs[0]?.id || ''));
                setJobsError('');
                setJobsRefreshed(new Date().toLocaleTimeString());
            } catch (error) {
                if (!controller.signal.aborted) setJobsError(errorText(error));
            } finally {
                if (!controller.signal.aborted) {
                    setJobsLoading(false);
                    timer = window.setTimeout(refresh, 2000);
                }
            }
        }
        void refresh();
        return () => { controller.abort(); clearTimeout(timer); };
    }, [refreshEpoch]);

    useEffect(() => {
        setDetail(null);
        setApproveOpen(false);
        setApproveNote('');
        setReport(null);
        setReportOpen(false);
        setArtifactsOpen(false);
        setLogsOpen(false);
        setLogs('');
        setLogsTrimmed(false);
        setDetailError('');
        setReportError('');
        setLogsError('');
        setStageId('');
        setActionError('');
        if (!selectedId) return;
        const controller = new AbortController();
        let timer: number | undefined;
        let offset = 0;
        let receivedCharacters = 0;
        setDetailLoading(true);
        async function refresh() {
            const results = await Promise.allSettled([
                processingApi.job(selectedId, controller.signal),
                processingApi.logs(selectedId, offset, controller.signal),
            ]);
            if (controller.signal.aborted) return;
            const [jobResult, logsResult] = results;
            if (jobResult.status === 'fulfilled') {
                setDetail(jobResult.value.job);
                setDetailError('');
            } else setDetailError(errorText(jobResult.reason));
            if (logsResult.status === 'fulfilled') {
                const chunk = logsResult.value;
                if (chunk.next_offset < offset) {
                    receivedCharacters = chunk.text.length;
                    setLogs(chunk.text.slice(-LOG_LIMIT));
                } else if (chunk.text) {
                    receivedCharacters += chunk.text.length;
                    setLogs((previous) => previous.length + chunk.text.length > LOG_LIMIT
                        ? (previous + chunk.text).slice(-LOG_LIMIT)
                        : previous + chunk.text);
                }
                setLogsTrimmed(receivedCharacters > LOG_LIMIT);
                offset = chunk.next_offset;
                setLogsError('');
            } else setLogsError(errorText(logsResult.reason));
            setDetailLoading(false);
            timer = window.setTimeout(refresh, 2000);
        }
        void refresh();
        return () => { controller.abort(); clearTimeout(timer); };
    }, [selectedId]);

    useEffect(() => {
        if (!selectedId || (!reportOpen && !stageId && !detail)) return;
        const controller = new AbortController();
        let timer: number | undefined;
        async function refreshReport() {
            try {
                const result = await processingApi.report(selectedId, controller.signal);
                if (controller.signal.aborted) return;
                setReport(result.report);
                setReportError('');
            } catch (error) {
                if (!controller.signal.aborted) setReportError(errorText(error));
            } finally {
                if (!controller.signal.aborted) timer = window.setTimeout(refreshReport, 2000);
            }
        }
        void refreshReport();
        return () => { controller.abort(); clearTimeout(timer); };
    }, [selectedId, reportOpen, stageId, detail?.id]);

    useEffect(() => {
        if (followLogs && logsOpen && logsContainer.current) logsContainer.current.scrollTop = logsContainer.current.scrollHeight;
    }, [logs, followLogs, logsOpen]);

    const selectedPaths = Object.keys(selectedInputs);
    const job = detail?.id === selectedId ? detail : jobs.find((entry) => entry.id === selectedId) ?? null;
    const selectedStage = job?.stages.find((entry) => entry.id === stageId) ?? null;
    const stageArtifacts = selectedStage ? job?.artifacts.filter((entry) => entry.stage_id === selectedStage.id
        || (!!selectedStage.checkpoint && entry.path === selectedStage.checkpoint)) ?? [] : [];
    const selectedSnapshot = selectedStage && job ? stageSnapshot(report, selectedStage, job.stages) : null;
    const humanNames = useMemo(() => {
        const rows = report && Array.isArray(report.subjects) ? (report.subjects as ReportRow[]) : [];
        return new Set(rows.filter((row) => row.rig_class === 'HUMAN_RIG' && row.name).map((row) => String(row.name)));
    }, [report]);
    const stageGroups = useMemo(() => (job ? buildStageGroups(job.stages) : []), [job?.stages]);
    const stageTopDone = stageGroups.filter((group) => group.kind === 'single'
        ? ['SUCCESS', 'FAILED', 'SKIPPED'].includes(group.stage.status)
        : group.items.every((item) => ['SUCCESS', 'FAILED', 'SKIPPED'].includes(item.stage.status))).length;
    const stageInspectBox = selectedStage ? <div className="stage-inspect">
        <div className="col-head"><h3>阶段详情 · {selectedStage.label || selectedStage.name}</h3><span className="grow" />
            {stageArtifacts.map((artifact) => <ArtifactLink key={artifact.id} artifact={artifact} />)}</div>
        {selectedStage.checkpoint && <p className="path hint" title={selectedStage.checkpoint}>检查点：{selectedStage.checkpoint}</p>}
        {selectedSnapshot ? <div className="raw"><JsonDiagnostic value={{ stage: selectedStage, report_snapshot: selectedSnapshot }} /></div>
            : <p className="hint">该阶段尚无报告快照。</p>}
    </div> : null;
    const stageRow = (stage: ProcessingStage, number: number | null) => <li key={stage.id}>
        <button type="button" className={`stage-row tone-${statusClass(stage.status)} ${stage.id === stageId ? 'selected' : ''}`}
            aria-pressed={stage.id === stageId} title={stageTitle(stage)}
            onClick={(event) => {
                const next = stage.id === stageId ? '' : stage.id;
                setStageId(next);
                if (next) { const el = event.currentTarget; requestAnimationFrame(() => el.scrollIntoView({ block: 'nearest' })); }
            }}>
            {number != null && <span className="stage-index">{number}</span>}
            <span className="stage-main"><strong>{stage.label || stage.name}{stage.subject ? ` · ${stage.subject}` : ''}</strong>
                <span className="stage-meta">{shortTime(stage.started_at)} → {shortTime(stage.finished_at)}</span></span>
            <span className={`qi-state ${statusClass(stage.status)}`}>{stageStatusText(stage.status)}{stage.elapsed_seconds != null ? ` · ${stage.elapsed_seconds.toFixed(1)}s` : ''}</span>
        </button>
        {stage.error && <p className="callout stage-error">{stage.error}</p>}
        {stage.id === stageId && stageInspectBox}
    </li>;
    const optionsValid = jitter.trim() !== '' && Number.isFinite(Number(jitter)) && Number(jitter) >= 0
        && tolerance.trim() !== '' && Number.isFinite(Number(tolerance)) && Number(tolerance) > 0;
    const nativeUnavailable = !!config && !config.shogun_available && selectedPaths.some((entry) => selectedInputs[entry].kind !== 'fbx');
    const createDisabled = !config || creating || !selectedPaths.length || !optionsValid || nativeUnavailable;
    function relSegments(path: string): string[] {
        for (const root of config?.data_roots || []) {
            if (path.toLowerCase().startsWith(root.toLowerCase() + '\\')) return path.slice(root.length + 1).split(/[\\/]+/);
        }
        return path.split(/[\\/]+/);
    }
    function isInputTreeDir(path: string): boolean {
        const first = relSegments(path)[0] || '';
        return !/^(02_|03_|_)/i.test(first);
    }
    async function addDirectory(path: string) {
        try {
            const result = await processingApi.browse(path, true);
            const processed = new Set(jobs.map((entry) => entry.input_path.toLowerCase()));
            let archived = new Set<string>();
            try {
                archived = new Set((await processingApi.outputs(500)).outputs.map((entry) => entry.path.toLowerCase()));
            } catch { /* 归档信息不可用时不做归档过滤 */ }
            const added: { path: string; name: string; kind: InputKind }[] = [];
            let skipped = 0;
            for (const seq of result.sequences) {
                const lower = seq.path.toLowerCase();
                if (processed.has(lower)) { skipped++; continue; }
                const rel = relSegments(seq.path).join('\\').toLowerCase();
                const stem = rel.replace(/\.(mcp|vdf|fbx)$/, '');
                if (archived.has(`02_processed\\${stem}.fbx`)) { skipped++; continue; }
                added.push({ path: seq.path, name: seq.name, kind: seq.kind });
            }
            const dirname = path.split(/[\\/]+/).filter(Boolean).pop() || path;
            if (!added.length) { setNotice(`目录「${dirname}」没有未处理序列${skipped ? `(已处理跳过 ${skipped} 条)` : ''}。`); return; }
            if (added.length > 30 && !window.confirm(`加入目录「${dirname}」的 ${added.length} 条新序列(跳过已处理 ${skipped} 条),继续?`)) return;
            setSelectedInputs((previous) => {
                const next = { ...previous };
                for (const entry of added) next[entry.path] = { name: entry.name, kind: entry.kind };
                return next;
            });
            setNotice(`已加入目录「${dirname}」:新增 ${added.length} 条序列,跳过已处理 ${skipped} 条。`);
        } catch (error) { setActionError(errorText(error)); }
    }
    const fbxSegs = useMemo(() => (fbxFiles || [])
        .filter((entry) => /\.fbx$/i.test(entry.name))
        .map((entry) => ({ entry, segs: relSegments(entry.path) })), [fbxFiles, config]);
    const cascade = useMemo(() => {
        const levels: { depth: number; options: string[] }[] = [];
        let d = 0;
        while (d <= chain.length) {
            const prefix = chain.slice(0, d);
            const opts = new Set<string>();
            for (const item of fbxSegs) {
                if (item.segs.length <= d + 1) continue;
                if (prefix.every((c, i) => item.segs[i] === c)) opts.add(item.segs[d]);
            }
            if (!opts.size) break;
            levels.push({ depth: d, options: [...opts].sort() });
            if (chain.length <= d) break;
            d += 1;
        }
        return levels;
    }, [fbxSegs, chain]);
    const fileOptions = useMemo(() => fbxSegs
        .filter((item) => item.segs.length === chain.length + 1 && chain.every((c, i) => item.segs[i] === c))
        .map((item) => item.entry)
        .sort((a, b) => a.name.localeCompare(b.name, 'zh-CN', { numeric: true })), [fbxSegs, chain]);

    function loadOutputs() {
        processingApi.outputs(100).then((result) => setFbxFiles(result.outputs)).catch(() => setFbxFiles([]));
    }
    function pickChainLevel(depth: number, value: string) {
        const next = chain.slice(0, depth);
        if (value) next[depth] = value;
        setChain(next);
    }
    function loadViewerUrl(fileUrl: string, name: string, bytes: number) {
        if (!config) return;
        if (bytes > LARGE_FILE
            && !window.confirm(`${name} 为 ${sizeText(bytes)}（超过 128 MB），加载会占用较多内存。确认加载？`)) return;
        const url = new URL(config.viewer_base, window.location.origin);
        url.searchParams.set('embed', '1');
        url.searchParams.set('file', fileUrl);
        url.searchParams.set('name', name);
        setViewerError('');
        setViewerLoading(true);
        setViewerSource(url.pathname + url.search);
    }
    function pickFile(entry: OutputEntry) {
        setCurrentPath(entry.path);
        loadViewerUrl(`/api/processing/output-file?path=${encodeURIComponent(entry.path)}`, entry.name, entry.size_bytes);
    }
    function inspectInViewer(artifact: ProcessingArtifact) {
        setView('viewer');
        if (fbxFiles === null) loadOutputs();
        let target = artifact;
        const firstSeg = relSegments(artifact.path)[0] || '';
        if (!/^(02_|03_)/i.test(firstSeg)) {
            // 任务目录里的完整工作版:优先映射到 03_exports 的同名可用版
            const base = (artifact.name || '').replace(/\.fbx$/i, '');
            const twin = (fbxFiles || []).find((entry) =>
                /^03_/i.test(relSegments(entry.path)[0] || '') && entry.name.replace(/\.fbx$/i, '') === base);
            if (twin) target = { ...artifact, path: twin.path, name: twin.name };
        }
        const segs = relSegments(target.path);
        setChain(segs.length > 1 ? segs.slice(0, -1) : []);
        setCurrentPath(target.path);
        const archived = /^(02_|03_)/i.test(segs[0] || '');
        loadViewerUrl(archived ? `/api/processing/output-file?path=${encodeURIComponent(target.path)}` : target.url,
                      target.name, target.size_bytes);
    }

    function toggleDirectory(path: string) {
        setExpanded((previous) => {
            const next = { ...previous };
            if (next[path]) delete next[path];
            else next[path] = true;
            return next;
        });
        if (!treeData[path]) loadTree(path);
    }

    function setSelected(entry: SelectedInput, path: string, on: boolean) {
        setSelectedInputs((previous) => {
            const next = { ...previous };
            if (on) next[path] = entry;
            else delete next[path];
            return next;
        });
    }

    // 一个搜索框两种模式：@ 前缀 → 跨数据根模糊搜索；否则过滤已加载层级。
    const mentionMode = searchText.startsWith('@');
    const mxFilter = mentionMode ? '' : searchText;

    function stopMentionSearch() {
        window.clearTimeout(mentionTimer.current);
        mentionController.current?.abort();
        setMention(null);
        setMentionBusy(false);
        setMentionError('');
    }

    function runFind(query: string, delay = 180) {
        setMentionBusy(true);
        window.clearTimeout(mentionTimer.current);
        mentionTimer.current = window.setTimeout(() => {
            const controller = new AbortController();
            mentionController.current = controller;
            processingApi.find(query, controller.signal).then((result) => {
                if (controller.signal.aborted) return;
                setMention(result);
                setMentionIndex(0);
                setMentionError('');
            }).catch((error: unknown) => {
                if (!controller.signal.aborted) { setMention(null); setMentionError(errorText(error)); }
            }).finally(() => {
                if (!controller.signal.aborted) setMentionBusy(false);
            });
        }, delay);
    }

    function onSearchChange(value: string) {
        setSearchText(value);
        if (value.startsWith('@') && value.slice(1).trim()) runFind(value.slice(1).trim());
        else stopMentionSearch();
    }

    function pickMention(match: FindMatch) {
        setSelected({ name: match.name, kind: match.kind }, match.path, true);
        setSearchText('');
        setMention(null);
        setMentionError('');
    }

    function onSearchKeyDown(event: ReactKeyboardEvent<HTMLInputElement>) {
        const matches = mention?.matches ?? [];
        if (mentionMode && matches.length) {
            if (event.key === 'ArrowDown') { event.preventDefault(); setMentionIndex((i) => (i + 1) % matches.length); return; }
            if (event.key === 'ArrowUp') { event.preventDefault(); setMentionIndex((i) => (i - 1 + matches.length) % matches.length); return; }
            if (event.key === 'Escape') { event.preventDefault(); setMention(null); return; }
            if (event.key === 'Enter') { event.preventDefault(); pickMention(matches[Math.min(mentionIndex, matches.length - 1)]); return; }
        }
        if (event.key === 'Enter') {
            event.preventDefault();
            const query = mentionMode ? searchText.slice(1).trim() : searchText.trim();
            if (!query) return;
            const controller = new AbortController();
            mentionController.current?.abort();
            mentionController.current = controller;
            setMentionBusy(true);
            processingApi.find(query, controller.signal).then((result) => {
                if (controller.signal.aborted) return;
                if (result.matches.length === 1) pickMention(result.matches[0]);
                else { setMention(result); setSearchText(`@${query}`); }
            }).catch((error: unknown) => { if (!controller.signal.aborted) setMentionError(errorText(error)); })
                .finally(() => { if (!controller.signal.aborted) setMentionBusy(false); });
        }
    }

    async function createJobs(mode: 'manual' | 'auto') {
        if (createDisabled || !config) return;
        setCreating(true);
        setActionError('');
        setNotice('');
        const options: ProcessingOptions = {
            filter_markers: optionFlags.includes('filter_markers'),
            dataset_mode: optionFlags.includes('dataset_mode'),
            attach_meshes: optionFlags.includes('attach_meshes'),
            lightweight: optionFlags.includes('lightweight'),
            drop_unused_labeling: optionFlags.includes('drop_unused_labeling'),
            denoise_static: optionFlags.includes('denoise_static'),
            expected_subjects: null, expected_props: null,
            prop_jitter_max_correction: Number(jitter), prop_tolerance: Number(tolerance),
        };
        try {
            const result = await processingApi.createJobs(selectedPaths, mode, options, config.csrf_token);
            setJobs((previous) => [...result.jobs, ...previous.filter((entry) => !result.jobs.some((created) => created.id === entry.id))]);
            if (result.jobs[0]) setSelectedId(result.jobs[0].id);
            setSelectedInputs({});
            setNotice(`已创建 ${result.jobs.length} 个${mode === 'manual' ? '逐步' : '自动'}任务；每次运行使用独立目录，不覆盖原始文件。`);
            setRefreshEpoch((previous) => previous + 1);
        } catch (error) {
            setActionError(errorText(error));
            setRefreshEpoch((previous) => previous + 1);
        } finally { setCreating(false); }
    }

    async function act(action: JobAction) {
        if (!job || !config || pendingAction) return;
        const id = job.id;
        setPendingAction(action);
        setActionError('');
        setNotice('');
        try {
            const result = await processingApi.action(id, action, config.csrf_token);
            setJobs((previous) => {
                const other = previous.filter((entry) => entry.id !== result.job.id);
                return [result.job, ...other];
            });
            if (result.job.id !== id) setSelectedId(result.job.id);
            else setDetail(result.job);
            setNotice(action === 'pause' ? '暂停请求已提交；当前原生调用完成后才会暂停。'
                : action === 'preview' ? '已请求导出当前真实场景；请等待产物列表更新。'
                    : action === 'retry' ? '已创建新运行；原运行及诊断保留。' : '操作请求已提交。');
        } catch (error) {
            setActionError(errorText(error));
            try {
                const result = await processingApi.job(id);
                setDetail((previous) => previous?.id === id ? result.job : previous);
            } catch { /* The original actionable error remains visible. */ }
        } finally {
            setPendingAction(null);
            setRefreshEpoch((previous) => previous + 1);
        }
    }

    async function saveWorkers(value?: string) {
        if (!config) return;
        const wanted = value ?? workers;
        setSavingWorkers(true);
        setConfigError('');
        try {
            const result = await processingApi.setWorkers(Number(wanted), config.csrf_token);
            setConfig(result);
            setWorkers(String(result.max_workers));
            setNotice(`并行上限已保存(${result.max_workers});等待中的原生会话仍占用执行 / 许可槽位。`);
        } catch (error) { setConfigError(errorText(error)); }
        finally { setSavingWorkers(false); }
    }

    async function approveJob(note: string) {
        if (!config || !job) return;
        setApproveOpen(false);
        setPendingAction('approve');
        try {
            const result = await processingApi.action(job.id, 'approve', config.csrf_token, note.trim());
            setDetail(result.job);
            setNotice(`已放行「${result.job.name}」:状态改为完成,放行记录已写入报告。`);
        } catch (error) { setActionError(errorText(error)); }
        finally { setPendingAction(null); setApproveNote(''); }
    }

    async function removeJob(entry: ProcessingJob) {
        if (!config || !TERMINAL[entry.state]) return;
        if (!window.confirm(`移除任务 ${entry.name}?该任务的报告、日志、检查点与产物会从任务存储删除;原始输入文件不受影响。不可撤销。`)) return;
        try {
            await processingApi.deleteJob(entry.id, config.csrf_token);
            setNotice(`已移除任务 ${entry.name}。`);
            if (selectedId === entry.id) setSelectedId('');
        } catch (error) { setActionError(errorText(error)); }
        finally { setRefreshEpoch((previous) => previous + 1); }
    }

    async function clearFinished() {
        if (!config) return;
        const finished = jobs.filter((entry) => TERMINAL[entry.state]).length;
        if (!finished) return;
        if (!window.confirm(`清空 ${finished} 个已结束任务?其报告、日志、检查点与产物都会被删除;活动与排队任务不受影响。不可撤销。`)) return;
        try {
            const result = await processingApi.clearFinished(config.csrf_token);
            setNotice(`已清空 ${result.count} 个已结束任务。`);
            if (result.removed.includes(selectedId)) setSelectedId('');
        } catch (error) { setActionError(errorText(error)); }
        finally { setRefreshEpoch((previous) => previous + 1); }
    }

    async function clearArchive() {
        if (!config || !job) return;
        const count = job.artifacts.filter((entry) => entry.role.startsWith('archive_')).length;
        if (!count) return;
        if (!window.confirm(`删除本运行归档到数据树的 ${count} 个副本(02_processed / 03_exports 下的 fbx/vdf/vsk)?\n任务目录、源数据与其他产物不受影响;再次运行会重新归档。不可撤销。`)) return;
        try {
            const result = await processingApi.clearArchive(job.id, config.csrf_token);
            setNotice(`已清除 ${result.removed.length} 个归档副本。`);
            setRefreshEpoch((previous) => previous + 1);
        } catch (error) { setActionError(errorText(error)); }
    }

    const controlDisabled = !config || !!pendingAction || !job || detailLoading || !!detailError;
    const counts: { QUEUED: number; RUNNING: number; WAITING: number; NEEDS_REVIEW: number; SUCCESS: number; PROBLEM: number } =
        { QUEUED: 0, RUNNING: 0, WAITING: 0, NEEDS_REVIEW: 0, SUCCESS: 0, PROBLEM: 0 };
    for (const entry of jobs) {
        if (entry.state === 'QUEUED') counts.QUEUED += 1;
        else if (entry.state === 'RUNNING' || entry.state === 'STARTING') counts.RUNNING += 1;
        else if (entry.state === 'WAITING' || entry.state === 'PAUSE_REQUESTED') counts.WAITING += 1;
        else if (entry.state === 'NEEDS_REVIEW') counts.NEEDS_REVIEW += 1;
        else if (entry.state === 'SUCCESS') counts.SUCCESS += 1;
        else counts.PROBLEM += 1;
    }

    const KIND_LETTER: Record<InputKind, string> = { mcp: 'M', vdf: 'V', fbx: 'F' };

    function filteredEntries(level: TreeResult) {
        const needle = mxFilter.trim().toLowerCase();
        if (!needle) return level.entries;
        return level.entries.filter((entry) => entry.name.toLowerCase().includes(needle) || (entry.kind === 'dir' && !!expanded[entry.path]));
    }

    function visibleRows() {
        const rows: { path: string; isDir: boolean; depth: number; parent: string | null }[] = [];
        const walk = (level: TreeResult, depth: number, parent: string | null) => {
            for (const entry of filteredEntries(level)) {
                rows.push({ path: entry.path, isDir: entry.kind === 'dir', depth, parent });
                if (entry.kind === 'dir' && expanded[entry.path] && treeData[entry.path]) walk(treeData[entry.path], depth + 1, entry.path);
            }
        };
        if (rootTree) walk(rootTree, 0, null);
        return rows;
    }

    function moveFocus(delta: number, path: string) {
        const rows = visibleRows();
        const index = rows.findIndex((row) => row.path === path);
        const next = rows[Math.min(Math.max(index + delta, 0), rows.length - 1)];
        if (next) treeRefs.current.get(next.path)?.focus();
    }

    function treeKeyDown(event: ReactKeyboardEvent<HTMLButtonElement>, path: string, isDir: boolean) {
        const rows = visibleRows();
        const current = rows.find((row) => row.path === path);
        switch (event.key) {
            case 'ArrowDown': event.preventDefault(); moveFocus(1, path); break;
            case 'ArrowUp': event.preventDefault(); moveFocus(-1, path); break;
            case 'Home': event.preventDefault(); treeRefs.current.get(rows[0]?.path ?? '')?.focus(); break;
            case 'End': event.preventDefault(); treeRefs.current.get(rows[rows.length - 1]?.path ?? '')?.focus(); break;
            case 'ArrowRight':
                event.preventDefault();
                if (isDir && !expanded[path]) toggleDirectory(path);
                else moveFocus(1, path);
                break;
            case 'ArrowLeft': {
                event.preventDefault();
                if (isDir && expanded[path]) toggleDirectory(path);
                else if (current?.parent) treeRefs.current.get(current.parent)?.focus();
                break;
            }
        }
    }

    function stageTitle(stage: ProcessingStage): string | undefined {
        const declared = config?.stages.find((entry) => entry.id === stage.id || entry.label === (stage.label ?? stage.name));
        return declared ? `${declared.label}：${declared.description}` : undefined;
    }

    const jobByPath = useMemo(() => {        const map = new Map<string, ProcessingJob>();
        for (const entry of jobs) {
            const current = map.get(entry.input_path);
            if (!current) { map.set(entry.input_path, entry); continue; }
            const currentActive = !TERMINAL[current.state];
            const entryActive = !TERMINAL[entry.state];
            if (entryActive === currentActive ? entry.created_at > current.created_at : entryActive) map.set(entry.input_path, entry);
        }
        return map;
    }, [jobs]);
    function renderTree(level: TreeResult, depth: number) {
        const entries = filteredEntries(level);
        return entries.map((entry, index) => {
            const pad = { ['--depth-pad' as string]: `${depth * 14 + 6}px` } as CSSProperties;
            if (entry.kind === 'dir') {
                const open = !!expanded[entry.path];
                const loading = !!treeBusy[entry.path];
                const error = treeErrors[entry.path];
                return <div key={entry.path}>
                    <button type="button" className={`tree-row${open ? ' open' : ''}`} data-dir={entry.path} style={pad}
                        role="treeitem" aria-expanded={open} aria-level={depth + 1} aria-posinset={index + 1} aria-setsize={entries.length}
                        aria-busy={loading || undefined} ref={(element) => { if (element) treeRefs.current.set(entry.path, element); else treeRefs.current.delete(entry.path); }}
                        onClick={() => toggleDirectory(entry.path)} onKeyDown={(event) => treeKeyDown(event, entry.path, true)}>
                        <span className="tw" aria-hidden="true" />
                        <span className="fname">{entry.name}</span>
                        {loading && <span className="fstat" role="status">…</span>}
                        {error && <span className="fstat warn" title={error}>!</span>}
                        {isInputTreeDir(entry.path) && <span className="fop">
                            <button type="button" className="tree-fop" title="加入整个目录:自动收集未处理序列,跳过已处理"
                                aria-label={`加入目录 ${entry.name}`}
                                onClick={(event) => { event.stopPropagation(); void addDirectory(entry.path); }}>
                                <i className="ico ico-plus" aria-hidden="true" />
                            </button>
                        </span>}
                    </button>
                    <div className="tree-children" role="group">
                        {open && (treeData[entry.path]
                            ? renderTree(treeData[entry.path], depth + 1)
                            : loading
                                ? <div className="tree-row muted" style={{ ['--depth-pad' as string]: `${(depth + 1) * 14 + 6}px` }} role="status"><span className="tw" /><span className="fname">读取中…</span></div>
                                : null)}
                    </div>
                </div>;
            }
            const on = !!selectedInputs[entry.path];
            return <button key={entry.path} type="button" className={`tree-row${on ? ' is-selected' : ''}`} data-file={entry.path} style={pad}
                role="treeitem" aria-pressed={on} aria-level={depth + 1} aria-posinset={index + 1} aria-setsize={entries.length}
                title={`${entry.path} · ${sizeText(entry.size_bytes)} · 点击加入待处理,提交后第一阶段即解析名单`}
                ref={(element) => { if (element) treeRefs.current.set(entry.path, element); else treeRefs.current.delete(entry.path); }}
                onClick={() => setSelected({ name: entry.name, kind: entry.kind }, entry.path, !on)} onKeyDown={(event) => treeKeyDown(event, entry.path, false)}>
                <span className="tw" />
                <span className="fname">{entry.name}</span>
                {entry.kind === 'mcp' && !entry.x2d_exists && <span className="fstat warn" title="未找到同名 X2D，离线重建可能失败">!</span>}
                <span className={`fstat k-${entry.kind}`}><span aria-hidden="true">{KIND_LETTER[entry.kind]}</span><span className="sr-only"> {entry.kind.toUpperCase()}</span></span>
                <span className="fstat check">{on ? '已选' : ''}</span>
                {jobByPath.has(entry.path) && (() => {
                    const owner = jobByPath.get(entry.path)!;
                    return <button type="button" className={`chip job-chip ${statusClass(owner.state)}`}
                        title={`任务 ${owner.id} · ${STATE_LABELS[owner.state] ?? owner.state}(点击在队列中打开)`}
                        onClick={(event) => { event.stopPropagation(); setSelectedId(owner.id); }}>#{owner.id.slice(0, 8)}</button>;
                })()}
                <span className="fop">
                    <button type="button" className="tree-fop" aria-label={`${on ? '从待处理清单移除' : '在任务详情打开序列解析'} ${entry.name}`}
                        onClick={(event) => { event.stopPropagation(); openInspect(entry.path); } }>
                        <i className="ico ico-search" aria-hidden="true" />
                    </button>
                    {entry.deletable && <button type="button" className="tree-fop del" aria-label={`删除处理产出 ${entry.name}`}
                        title="删除该处理产出文件(仅 02_*/03_* 结果目录;不可撤销)"
                        onClick={(event) => { event.stopPropagation(); void deleteOutputFile(entry); }}>
                        <i className="ico ico-trash" aria-hidden="true" />
                    </button>}
                </span>
            </button>;
        });
    }

    const rootTree = treeData[rootPath];
    const mentionPanelOpen = mentionMode && (mentionBusy || !!mention || !!mentionError);
    const inspectFps = inspectData ? (inspectData.frame_rate_hz ?? inspectData.capture_clock?.frequency_hz ?? null) : null;
    const bestFbxArtifact = useMemo(() => (job ? viewerPreferred(job.artifacts) : null), [job?.artifacts]);
    const inspectSelected: InspectItem | null = inspectData && inspectItemKey.includes('|')
        ? ((inspectItemKey.startsWith('s|') ? inspectData.subjects : inspectData.objects).find((item) => item.name === inspectItemKey.slice(2)) ?? null)
        : null;

    function rosterButton(group: 's' | 'o', item: InspectItem) {
        const key = `${group}|${item.name}`;
        return <button key={key} type="button" className={`roster-item${inspectItemKey === key ? ' selected' : ''}`}
            aria-pressed={inspectItemKey === key} onClick={() => setInspectItemKey(key)}>
            <strong>{item.name}</strong>
            <span className="roster-badges">
                {item.bones != null && <span title="骨骼 / LimbNode">骨 {item.bones}</span>}
                {item.markers != null && <span title="标记点数">mk {item.markers}</span>}
                {item.meshes != null && item.meshes > 0 && <span title="网格数">网 {item.meshes}</span>}
                {item.rigid && <span title="MCP 内的刚体标记集定义">刚体</span>}
                {item.occurrences != null && <span title="文件内定义记录条数">×{item.occurrences}</span>}
                {item.rig_class === 'HUMAN_RIG' && <span className="green">人体</span>}
                {item.rig_class === 'PROP_OR_NONHUMAN' && <span>道具</span>}
                {item.sources.map((source) => <span key={source} className="src">{SOURCE_LABELS[source] ?? source}</span>)}
            </span>
        </button>;
    }

    return <main id="main-content" className="app" tabIndex={-1}>
        <header className="bar">
            <span className="brand"><span className="brand-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="22" height="22">
                <path d="M6 18 L12 6 L18 18 Z" fill="none" stroke="rgba(255,255,255,.9)" strokeWidth="1.6" strokeLinejoin="round" />
                <circle className="bd b1" cx="6" cy="18" r="2.7" fill="#ffe66d" />
                <circle className="bd b2" cx="12" cy="6" r="2.7" fill="#ffffff" />
                <circle className="bd b3" cx="18" cy="18" r="2.7" fill="#ff9ec6" />
              </svg>
            </span>Mocap Studio</span>
            <span className="count-chip" title="排队中">排队 <b>{counts.QUEUED}</b></span>
            <span className={`count-chip ${counts.RUNNING ? 'blue' : ''}`} title="执行中（含启动中）">执行 <b>{counts.RUNNING}</b></span>
            <span className={`count-chip ${counts.WAITING ? 'blue' : ''}`} title="等待操作 / 暂停中的任务">等待 <b>{counts.WAITING}</b></span>
            <span className={`count-chip ${counts.NEEDS_REVIEW ? 'amber' : ''}`} title="执行完成但数据质量需人工复核">复核 <b>{counts.NEEDS_REVIEW}</b></span>
            <span className={`count-chip ${counts.SUCCESS ? 'green' : ''}`} title="完成">完成 <b>{counts.SUCCESS}</b></span>
            <span className={`count-chip ${counts.PROBLEM ? 'red' : ''}`} title="失败 / 中断 / 取消">异常 <b>{counts.PROBLEM}</b></span>
            <div className="seg" role="tablist" aria-label="视图">
                <button type="button" className={view === 'home' ? 'active' : ''} onClick={() => setView('home')}>数据看板</button>
                <button type="button" className={view === 'work' ? 'active' : ''} onClick={() => setView('work')}>处理台</button>
                <button type="button" className={view === 'viewer' ? 'active' : ''} onClick={() => { setView('viewer'); if (fbxFiles === null) loadOutputs(); }}>FBX 查看</button>
            </div>
            <span className="grow" />
            {config && <>
                <span className={`chip ${config.shogun_available ? 'green' : 'amber'}`}
                    title={config.shogun_available ? `Shogun：${config.shogun_path}` : '未检测到 Shogun；MCP / VDF 不可处理，可处理现有 FBX'}>
                    Shogun {config.shogun_available ? "可用" : "缺失"}
                </span>
                <span className={`chip ${config.blender_path ? 'green' : 'faint'}`}
                    title={config.blender_path ? `Blender：${config.blender_path}` : '未检测到 Blender；网格挂载与部分轻量化可能失败'}>
                    Blender {config.blender_path ? "可用" : "缺失"}
                </span>
            </>}
            {metrics && metrics.cpu_percent != null && (
                <span className={`chip ${metrics.cpu_percent >= 85 ? 'red' : metrics.cpu_percent >= 60 ? 'amber' : 'green'}`}
                    title={`CPU 使用率 ${Math.round(metrics.cpu_percent)}%(全部核心合计;每 2.5s 刷新)`}>
                    CPU {Math.round(metrics.cpu_percent)}%
                </span>
            )}
            {metrics && metrics.memory_total_bytes && metrics.memory_used_bytes != null && (() => {
                const percent = Math.round(100 * metrics.memory_used_bytes! / metrics.memory_total_bytes!);
                return <span className={`chip ${percent >= 85 ? 'red' : percent >= 60 ? 'amber' : 'green'}`}
                    title={`内存 ${gbytesT(metrics.memory_used_bytes!)} / ${gbytesT(metrics.memory_total_bytes!)}(${percent}%)`}>
                    内存 {percent}%
                </span>;
            })()}
            {metrics && metrics.disk[0] && (
                <span className={`chip ${metrics.disk[0].free_bytes < 1024 ** 4 ? 'amber' : 'faint'}`}
                    title={metrics.disk.map((entry) => `${entry.path} · 剩 ${gbytesT(entry.free_bytes)} / 共 ${gbytesT(entry.total_bytes)}`).join('\n')}>
                    磁盘 {gbytesT(metrics.disk[0].free_bytes)} 可用
                </span>
            )}
            {configError && <button className="btn danger" type="button" onClick={() => setConfigEpoch((previous) => previous + 1)}>连接断开 · 重连</button>}
        </header>

        {configError && <div className="callout" role="alert">{configError}</div>}
        {actionError && <div className="callout" role="alert">{actionError}<button className="btn ghost" type="button" onClick={() => setActionError('')}>知道了</button></div>}
        {notice && <div className="callout info" role="status">{notice}</div>}

        {view === 'home' ? (
            <HomePanel config={config} jobs={jobs} counts={counts}
                onOpenJob={(id) => { setSelectedId(id); setView('work'); }}
                onInspectFile={(path) => { setView('work'); openInspect(path); }}
                onGoWork={() => setView('work')} />
        ) : view === 'viewer' ? (
        <section className="viewer-page" aria-label="FBX 查看器">
            <div className="viewer-page-bar">
                <button className="btn ghost" type="button" title="返回上一个页面(浏览器返回键同样有效)"
                    onClick={navigateBack}>← 返回</button>
                {cascade.map((level) => <label key={level.depth} className="field-inline">
                    <span aria-hidden="true">{level.depth === 0 ? '目录' : '›'}</span>
                    <select className="select" value={chain[level.depth] || ''}
                        onChange={(event) => pickChainLevel(level.depth, event.target.value)}>
                        <option value="">选择…</option>
                        {level.options.map((option) => <option key={option} value={option}>{option}</option>)}
                    </select></label>)}
                {!!fileOptions.length && <label className="field-inline"><span>文件</span>
                    <select className="select" value={currentPath} onChange={(event) => {
                        const entry = fileOptions.find((f) => f.path === event.target.value);
                        if (entry) pickFile(entry);
                    }}>
                        <option value="">选择 FBX…</option>
                        {fileOptions.map((entry) => <option key={entry.path} value={entry.path}>{entry.name.replace(/\.fbx$/i, '')} · {sizeText(entry.size_bytes)} · {shortTime(entry.modified_at)}</option>)}
                    </select></label>}
                <button className="btn ghost" type="button" disabled={!viewerSource} onClick={() => { setViewerSource(''); setCurrentPath(''); }}>卸载场景</button>
                {currentPath && <span className="hint mono viewer-path-label" title={currentPath}>{currentPath.replace(/^[A-Za-z]:[\\/]/, '')}</span>}
            </div>
            {fbxFiles === null && <p role="status" className="hint">扫描 02_processed / 03_exports 产出…</p>}
            {fbxFiles && !fbxSegs.length && !currentPath && <p className="wb-empty">结果目录中暂无 FBX 文件；运行完成后的可用版本会出现在这里。</p>}
            {viewerLoading && <p role="status" className="hint">正在加载；FBX 下载、解析与错误由查看器内部显示。</p>}
            {viewerError && <p className="callout" role="alert">{viewerError}</p>}
            {viewerSource ? <iframe key={viewerSource} className="viewer-frame viewer-frame-page" title="FBX 查看器"
                src={viewerSource} onLoad={() => setViewerLoading(false)} onError={() => { setViewerLoading(false); setViewerError('查看器页面加载失败，请检查本地服务。'); }} />
                : <p className="wb-empty">逐级选择目录后加载 FBX；从任务点「查看」会自动定位并直接加载对应文件。</p>}
        </section>
        ) : (
        <div className="cols" ref={colsRef}>
            {/* ── 左列：数据浏览器 + 待处理清单 ── */}
            <aside className="col col-explorer" aria-label="数据浏览器">
                <div className="root-row">
                    {config && <div className="root-chips" role="group" aria-label="数据根">
                        {config.data_roots.map((root) => <button key={root} type="button"
                            className={`btn ghost root-chip ${root === rootPath ? 'active' : ''}`} title={root}
                            onClick={() => { setRootPath(root); loadTree(root); }}>{root.split(/[\\/]+/).filter(Boolean).pop() || root}</button>)}
                    </div>}
                    <div className="search-wrap">
                        <i className="ico ico-search" aria-hidden="true" />
                        <input className="input" value={searchText} aria-label="搜索序列"
                            title="过滤当前已加载层级;输入 @ 可跨数据根搜索序列"
                            placeholder="搜索…"
                            onChange={(event) => onSearchChange(event.target.value)} onKeyDown={onSearchKeyDown} />
                        {mentionPanelOpen && <div className="cm-mention" role="listbox" aria-label="引用文件 / 序列">
                            <div className="cm-mention-head"><span>跨数据根搜索</span>
                                {mention ? <span className="muted tiny">{mention.query}</span> : null}</div>
                            {mentionBusy && <div className="cm-mention-empty muted small">正在搜索真实文件…</div>}
                            {mentionError && <div className="cm-mention-empty" role="alert">{mentionError}</div>}
                            {mention && !mention.matches.length && !mentionBusy && <div className="cm-mention-empty muted small">没有匹配项；搜索截断时请补全关键字。</div>}
                            {mention?.matches.map((match, index) => <button key={match.path} type="button" role="option" aria-selected={index === mentionIndex}
                                className={`cm-mention-item${index === mentionIndex ? ' active' : ''}`}
                                onMouseEnter={() => setMentionIndex(index)}
                                onMouseDown={(event) => { event.preventDefault(); pickMention(match); }}>
                                <span className={`fstat k-${match.kind}`} aria-hidden="true">{KIND_LETTER[match.kind]}</span>
                                <span className="cm-mention-label">{match.name}</span>
                                {match.kind === 'mcp' && !match.x2d_exists && <span className="fstat warn" title="未找到同名 X2D">!</span>}
                                <span className="cm-mention-detail">{match.path} · {sizeText(match.size_bytes)}</span>
                            </button>)}
                            {mention && <div className="cm-mention-foot muted tiny">扫描 {mention.scanned} 项 · 命中 {mention.matches.length}{mention.truncated ? ' · 已达上限截断' : ''} · ↑↓ 选择，回车加入</div>}
                        </div>}
                    </div>
                    <button className="btn ghost tree-refresh" type="button" title="重新读取目录树(含已展开层级)" aria-label="刷新目录树" onClick={refreshTree}>
                        <i className={`ico ico-refresh${Object.keys(treeBusy).length ? ' is-busy' : ''}`} aria-hidden="true" />
                    </button>
                </div>
                <div className="tree-box" role="tree" aria-label="数据目录树">
                    {treeErrors[rootPath || 'root'] && <p className="callout" role="alert">{treeErrors[rootPath || 'root']}</p>}
                    {rootTree ? renderTree(rootTree, 0) : <p className="wb-empty">{treeBusy[rootPath || 'root'] ? '读取目录…' : '尚未读取数据根。'}</p>}
                </div>
                <div className="tray-splitter" role="separator" aria-orientation="horizontal" aria-label="拖动调整处理区高度(↑↓ 键微调)"
                    tabIndex={0} onPointerDown={onTraySplitDown}
                    onKeyDown={(event) => {
                        if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
                            event.preventDefault();
                            const next = clampTray(trayHeight + (event.key === 'ArrowUp' ? -24 : 24));
                            setTrayHeight(next);
                            persistTray(next);
                        }
                    }} />
                <div className="tray-box" style={{ height: trayHeight }}>
                    <div className="pipeline">
                        <div className="pipeline-title">处理配置</div>
                        <AdvancedOptions flags={optionFlags} onFlags={setOptionFlags}
                            jitter={jitter} onJitter={setJitter} tolerance={tolerance} onTolerance={setTolerance} />
                        {optionFlags.includes('attach_meshes') && config && !config.blender_path && <p className="callout warn">未检测到 Blender；网格挂载可能失败，实际错误保留在任务日志与报告中。</p>}
                    </div>
                    <div className="tray-divider" role="presentation" />
                    <div className="create-row">
                        <button className="btn primary" type="button" aria-busy={creating} disabled={createDisabled || selectedPaths.length !== 1}
                            title="单条序列逐步处理：在真实阶段边界等待，每点一次执行一个阶段"
                            onClick={() => void createJobs('manual')}>逐步处理</button>
                        <button className="btn primary" type="button" disabled={createDisabled} aria-busy={creating}
                            title="自动批处理：连续执行整条管线；等待中的原生会话占用并行槽位"
                            onClick={() => void createJobs('auto')}>自动批处理{selectedPaths.length ? ` (${selectedPaths.length})` : ''}</button>
                        <span className="grow" />
                        <span className="field-inline" title="队列并发上限(1-8)，选择即自动保存;暂停中的原生会话仍占执行 / 许可槽位"><span>并发</span>
                            <WorkersSelect value={Number(workers)} max={config?.max_workers_limit ?? 8}
                                disabled={!config || savingWorkers}
                                onChange={(value) => { setWorkers(value); void saveWorkers(value); }} />
                        </span>
                    </div>
                    <div className="tray-divider" role="presentation" />
                    <div className="tray-meta">
                        <span>待处理 <b>{selectedPaths.length}</b> 条</span>
                        {rootTree?.truncated && <span className="hint" title="该层超过 500 项已截断，可用 @ 搜索定位">本层已截断</span>}
                        {selectedPaths.length > 0 && <button className="btn ghost" type="button" onClick={() => setSelectedInputs({})}>清空</button>}
                    </div>
                    {!!selectedPaths.length && <div className="tray">
                        {selectedPaths.map((entry) => <span key={entry} className="tray-chip" title={entry}>
                            <strong>{selectedInputs[entry].name}</strong>
                            <span className={`fstat k-${selectedInputs[entry].kind}`} aria-hidden="true">{KIND_LETTER[selectedInputs[entry].kind]}</span>
                            <button type="button" aria-label={`移除 ${entry}`} onClick={() => setSelected(selectedInputs[entry], entry, false)}>×</button>
                        </span>)}
                    </div>}
                    {nativeUnavailable && <p className="callout warn">所选 MCP / VDF 需要 Shogun；请恢复本地 SDK，或仅选现有 FBX。</p>}
                </div>
            </aside>

            {/* ── 中列：处理队列 ── */}
            <section className="col col-queue" aria-label="处理队列">
                <div className="col-head"><h2 title="每约 2 秒读取真实状态;当前会话由本机回环承载">处理队列</h2><span className="grow" />
                    {jobs.some((entry) => TERMINAL[entry.state]) && <button className="btn ghost" type="button"
                        title="移除全部已结束任务(完成/需复核/失败/取消/中断);活动与排队任务保留"
                        onClick={() => void clearFinished()}>清空已结束 ({jobs.filter((entry) => TERMINAL[entry.state]).length})</button>}
                    <button className="btn ghost" type="button" disabled={jobsLoading} onClick={() => setRefreshEpoch((previous) => previous + 1)}>刷新</button></div>
                {jobsError && <p className="callout" role="alert">{jobsError}（列表可能已过期）</p>}
                {jobsLoading && !jobs.length && <p role="status" className="hint">读取任务…</p>}
                {!jobsLoading && !jobs.length && !jobsError && <p className="wb-empty">尚无任务。在左侧选序列后创建运行。</p>}
                <div className="queue-list">{(() => {
                    const groups: { key: string; label: string; states: JobState[] }[] = [
                        { key: 'active', label: '执行中 / 排队', states: ['QUEUED', 'STARTING', 'RUNNING', 'WAITING', 'PAUSE_REQUESTED'] },
                        { key: 'review', label: '需人工复核', states: ['NEEDS_REVIEW'] },
                        { key: 'done', label: '完成', states: ['SUCCESS'] },
                        { key: 'bad', label: '失败 / 中断 / 取消', states: ['FAILED', 'INTERRUPTED', 'CANCELLED'] },
                    ];
                    const renderItem = (entry: ProcessingJob) => <button key={entry.id} type="button" className={`queue-item ${selectedId === entry.id ? 'selected' : ''}`}
                        aria-pressed={selectedId === entry.id} onClick={() => setSelectedId(entry.id)}>
                        <span className="qi-head"><strong className="qi-title">{entry.name}</strong>
                            <span className="qi-id mono" title={`任务 id:${entry.id}`}>#{entry.id.slice(0, 8)}</span>
                            <span className={`qi-state ${statusClass(entry.state)}`}>{STATE_LABELS[entry.state] ?? entry.state}</span>
                            {TERMINAL[entry.state] && <button type="button" className="qi-remove" title="移除该已结束任务(删除其报告/日志/检查点/产物)"
                                aria-label={`移除任务 ${entry.name}`}
                                onClick={(event) => { event.stopPropagation(); void removeJob(entry); }}><i className="ico ico-close" aria-hidden="true" /></button>}</span>
                        <span className="qi-meta">{entry.mode === 'manual' ? '逐步' : '自动'} · {entry.input_kind.toUpperCase()} · {shortTime(entry.created_at)}</span>
                    </button>;
                    return groups.map((group) => {
                        const items = jobs.filter((entry) => group.states.includes(entry.state));
                        if (!items.length) return null;
                        const open = !collapsed[group.key];
                        return <div key={group.key} className={`queue-group${open ? '' : ' closed'}`}>
                            <button type="button" className="queue-group-head" aria-expanded={open}
                                onClick={() => setCollapsed((previous) => ({ ...previous, [group.key]: open }))}>
                                <span className="tw" aria-hidden="true" />
                                <strong>{group.label}</strong>
                                <span className="count-chip">{items.length}</span>
                            </button>
                            {open && <div className="queue-group-body">{items.map(renderItem)}</div>}
                        </div>;
                    });
                })()}</div>
            </section>

            {/* ── 右列：运行详情（阶段直出，产物 / 日志 / 报告为折叠区）── */}
            <section className="col col-detail" aria-label="运行详情">
                {!job && <p className="wb-empty">{detailLoading ? '读取任务详情…' : '从队列选择一个运行。'}</p>}
                {detailError && job && <p className="callout" role="alert">{detailError}（详情可能已过期）</p>}
                {job && <>
                    <div className="detail-head">
                        <h2>{job.name}</h2>
                        <span className={`chip ${statusClass(job.state)}`}>{STATE_LABELS[job.state] ?? job.state}</span>
                        {job.quality_status && <span className="chip">质量 {job.quality_status}</span>}
                        {job.worker_pid && <span className="chip mono">PID {job.worker_pid}</span>}
                        {Array.isArray(job.options?.expected_subjects) && !!job.options.expected_subjects.length &&
                            <span className="chip" title={`预期人体名单(随报告记录): ${job.options.expected_subjects.join('、')}`}>预期人体 {job.options.expected_subjects.length}</span>}
                        {Array.isArray(job.options?.expected_props) && !!job.options.expected_props.length &&
                            <span className="chip" title={`预期物体名单(随报告记录): ${job.options.expected_props.join('、')}`}>预期物体 {job.options.expected_props.length}</span>}
                        <p className="path" title={job.input_path}>{job.input_path}</p>
                        {job.review && <p className="callout">已放行 · {shortTime(job.review.approved_at)}{job.review.note ? ` · ${job.review.note}` : ''}</p>}
                        {job.error && <p className="callout" role="alert">{job.error}</p>}
                    </div>
                    <div className="detail-actions">
                        <button className="btn primary" type="button" disabled={controlDisabled || job.state !== 'WAITING'} aria-busy={pendingAction === 'step'}
                            title="只执行当前等待的阶段；原生调用不可中途拆分"
                            onClick={() => void act('step')}>执行下一阶段</button>
                        <button className="btn" type="button" disabled={controlDisabled || job.state !== 'WAITING'} aria-busy={pendingAction === 'run'}
                        title="从此处连续执行剩余阶段" onClick={() => void act('run')}>连续执行</button>
                        <button className="btn" type="button" disabled={controlDisabled || job.state !== 'RUNNING'} aria-busy={pendingAction === 'pause'}
                            title="当前不可分割的原生调用完成后暂停" onClick={() => void act('pause')}>阶段后暂停</button>
                        <button className="btn" type="button" disabled={controlDisabled || job.state !== 'WAITING' || job.input_kind === 'fbx'} aria-busy={pendingAction === 'preview'}
                            title={job.input_kind === 'fbx' ? 'FBX 输入请直接到「产物」模块查看' : job.state !== 'WAITING' ? '导出预览 FBX 只在阶段间暂停时可用(需要 Shogun 会话存活);完成后的最终 FBX 请到「产物」模块点「查看」' : '导出当前真实场景的中间 FBX，可在查看器检查'} onClick={() => void act('preview')}>导出预览 FBX</button>
                        <button className="btn ghost" type="button" title="解析该输入文件的人物 / 物体名单"
                            onClick={() => openInspect(job.input_path)}>序列详情</button>
                        {bestFbxArtifact && <button className="btn outline" type="button"
                            title={`在 3D 查看器打开 ${bestFbxArtifact.name}(轻量/完整/逐人序列自动挑选)`}
                            onClick={() => inspectInViewer(bestFbxArtifact)}>查看结果 FBX</button>}
                        {job.state === 'NEEDS_REVIEW' && !approveOpen && <button className="btn primary" type="button" disabled={controlDisabled} aria-busy={pendingAction === 'approve'}
                            title="复核确认无问题:状态改为完成(放行时间与备注会写入报告)"
                            onClick={() => setApproveOpen(true)}>放行</button>}
                        <button className="btn" type="button" disabled={controlDisabled || !TERMINAL[job.state]} aria-busy={pendingAction === 'retry'}
                            title="从源文件新建运行；原运行及诊断保留，不从任意中间阶段假恢复" onClick={() => void act('retry')}>新建重试</button>
                        <button className="btn danger" type="button" disabled={controlDisabled || TERMINAL[job.state]} aria-busy={pendingAction === 'cancel'}
                            title="取消只回收本系统拥有的进程，保留检查点与诊断" onClick={() => void act('cancel')}>取消</button>
                    </div>
                    {job.state === 'NEEDS_REVIEW' && approveOpen && <div className="approve-strip">
                        <span className="hint">确认放行「{job.name}」?</span>
                        <input className="input" value={approveNote} maxLength={500} placeholder="备注(可留空)" autoFocus
                            onChange={(event) => setApproveNote(event.target.value)}
                            onKeyDown={(event) => {
                                if (event.key === 'Enter') void approveJob(approveNote);
                                if (event.key === 'Escape') { setApproveOpen(false); setApproveNote(''); }
                            }} />
                        <button className="btn primary" type="button" onClick={() => void approveJob(approveNote)}>确认放行</button>
                        <button className="btn ghost" type="button" onClick={() => { setApproveOpen(false); setApproveNote(''); }}>取消</button>
                    </div>}
                    <div className="meter-row">
                        <div className={`meter${!TERMINAL[job.state] ? ' indeterminate' : ''}`}>
                            {!TERMINAL[job.state]
                                ? <span className="meter-sweep" />
                                : <span style={{ width: `${stageGroups.length ? Math.round((stageTopDone / stageGroups.length) * 100) : 0}%` }} />}
                        </div>
                        <span className="hint" title={!TERMINAL[job.state] ? '运行中的管线长度会随名单展开,不显示伪完成度;结束后给出真实分数' : '按顶层阶段计数;逐人物/道具的子步骤合并为一个阶段'}>
                            {TERMINAL[job.state]
                                ? `${stageTopDone}/${stageGroups.length || 0} 阶段`
                                : `${job.stages.find((entry) => entry.id === job.current_stage)?.label || job.stages.find((entry) => entry.id === job.waiting_stage)?.label || (job.state === 'QUEUED' ? '排队中' : '启动中')}${job.state === 'WAITING' ? ' · 等待指令' : ' · 执行中'} · 已完成 ${stageTopDone} 阶段`}
                        </span>
                    </div>

                    {job.stages.length === 0 && <p className="wb-empty">尚无已回报阶段；状态以工作进程回报为准。</p>}
                    <ol className="stage-list" aria-label="真实阶段时间线">{(() => { let top = 0; return stageGroups.map((group) => group.kind === 'single' ? stageRow(group.stage, ++top) : (() => {
                        const number = ++top;
                        const open = !!groupOpen[group.key];
                        const summary = groupSummary(group.items);
                        const humans = group.items.filter((item) => humanNames.has(String(item.stage.subject)));
                        const objects = group.items.filter((item) => !humanNames.has(String(item.stage.subject)));
                        const split = humanNames.size > 0 && humans.length > 0 && objects.length > 0;
                        return <li key={group.key} className="stage-group">
                            <button type="button" className={`stage-row group${open ? ' open' : ''}`} aria-expanded={open}
                                onClick={() => setGroupOpen((previous) => ({ ...previous, [group.key]: !open }))}>
                                <span className="stage-index">{number}</span>
                                <span className="tw" aria-hidden="true" />
                                <span className="stage-main"><strong>{group.label} ({group.items.length})</strong>
                                    <span className="stage-meta">{split ? `${humans.length} 人物 · ${objects.length} 物体 · ` : ''}共 {summary.seconds.toFixed(1)}s</span></span>
                            <span className={`qi-state ${summary.tone}`}>{summary.text}</span>
                            </button>
                            {open && <ol className="stage-sub">{split ? <>
                                <li className="stage-subhead">人物（{humans.length}）</li>
                                {humans.map((item) => stageRow(item.stage, null))}
                                <li className="stage-subhead">物体（{objects.length}）</li>
                                {objects.map((item) => stageRow(item.stage, null))}
                            </> : group.items.map((item) => stageRow(item.stage, null))}</ol>}
                        </li>;
                    })()); })()}</ol>
                    <ReportSummary report={report} />
                    <ModuleCard title="处理配置" open={optionsOpen} onToggle={setOptionsOpen}>
                        <div className="raw"><JsonDiagnostic value={job.options} /></div>
                    </ModuleCard>

                    <ModuleCard title="产物" count={job.artifacts.length} tone={job.artifacts.length ? 'blue' : 'faint'}
                        open={artifactsOpen} onToggle={setArtifactsOpen}>
                        {job.artifacts.some((entry) => entry.role.startsWith('archive_')) && <div className="artifact-row">
                            <button className="btn outline" type="button"
                                title="删除本运行复制到 02_processed / 03_exports 的归档副本(任务目录与源数据不动)"
                                onClick={() => void clearArchive()}>清除归档副本 ({job.artifacts.filter((entry) => entry.role.startsWith('archive_')).length})</button>
                            <span className="hint">这些是归档到数据树的镜像;任务目录内的产物保留。</span>
                        </div>}
                        {job.artifacts.length === 0 && <p className="hint">尚无已注册产物；中间 FBX 需在等待时主动导出。</p>}
                        {job.artifacts.map((artifact) => <div key={artifact.id} className="artifact-row">
                            <ArtifactLink artifact={artifact} />
                            {isFbx(artifact) && <button className="btn outline" type="button" onClick={() => inspectInViewer(artifact)}>查看</button>}
                        </div>)}
                    </ModuleCard>

                    <ModuleCard title="日志" open={logsOpen} onToggle={setLogsOpen}>
                        <label className="feature-check"><input type="checkbox" checked={followLogs}
                            onChange={(event) => setFollowLogs(event.target.checked)} /><span>跟随新日志</span></label>
                        {logsError && <p className="callout" role="alert">{logsError}</p>}
                        <div className="raw logs" ref={logsContainer} tabIndex={0} aria-label="真实工作进程日志">
                            {logs ? <pre>{logs}</pre> : <p className="hint">尚无日志输出。</p>}
                        </div>
                        {logsTrimmed && <p className="hint">页面仅保留最近 524288 字符；完整日志在本地任务目录。</p>}
                    </ModuleCard>

                    <ModuleCard title="报告 JSON" open={reportOpen} onToggle={setReportOpen}>
                        {reportError && <p className="callout" role="alert">{reportError}</p>}
                        {report ? <div className="raw"><JsonDiagnostic value={report} /></div> : <p className="hint">按需读取实际报告…</p>}
                    </ModuleCard>
                </>}
            </section>
        </div>
        )}

        {/* ── 序列详情覆盖层：任务详情「序列详情」或树行悬停 ⌕ 打开 ── */}
        {inspectPath && <div className="overlay" role="dialog" aria-modal="true"
            aria-label={`序列详情 · ${inspectPath.split(/[\\/]/).pop() ?? inspectPath}`}
            onClick={(event) => { if (event.target === event.currentTarget) closeInspect(); }}>
            <div className="overlay-panel inspect-panel">
                <div className="col-head">
                    <h2>{inspectPath.split(/[\\/]/).pop()}</h2>
                    <span className="grow" />
                    <button className="btn outline" type="button"
                        onClick={() => setSelected({ name: inspectPath.split(/[\\/]/).pop() ?? inspectPath,
                            kind: inspectData?.kind ?? (inspectPath.endsWith('.vdf') ? 'vdf' : inspectPath.endsWith('.fbx') ? 'fbx' : 'mcp') }, inspectPath, !selectedInputs[inspectPath])}>
                        {selectedInputs[inspectPath] ? '移出待处理' : '加入待处理'}
                    </button>
                    <button className="btn ghost" type="button" onClick={closeInspect} aria-label="关闭序列详情"><i className="ico ico-close" aria-hidden="true" /></button>
                </div>
                {inspectData && <div className="inspect-facts inspect-meta">
                    <span className={`chip k-chip k-${inspectData.kind}`}>{inspectData.kind.toUpperCase()}</span>
                    <span className="chip faint">{sizeText(inspectData.size_bytes)}</span>
                    <span className="chip faint">{timeText(inspectData.modified_at)}</span>
                    {inspectData.dataset_family && <span className={`chip ${inspectData.dataset_family === 'MP' || inspectData.dataset_family === 'BODY' ? 'amber' : ''}`}
                        title={`数据族:${inspectData.dataset_family}${inspectData.dataset_family === 'MP' || inspectData.dataset_family === 'BODY' ? '(人体专用,处理时删除全部物体)' : ''}`}>
                        {inspectData.dataset_family}{inspectData.dataset_family === 'MP' || inspectData.dataset_family === 'BODY' ? ' · 人体专用' : ''}
                    </span>}
                    <span className={`chip ${inspectData.x2d_exists ? 'green' : 'faint'}`}
                        title={inspectData.x2d_exists ? '存在同名 X2D 离线重建源' : '没有同名 X2D;若需离线重建会失败'}>
                        X2D {inspectData.x2d_exists ? '✓' : '—'}
                    </span>
                    {inspectData.capture_clock?.frequency_hz != null && <span className="chip blue"
                        title={`捕获时钟(Input.ClockFrequency):${inspectData.capture_clock.frequency_hz} Hz${inspectData.capture_clock.timestamped ? ';帧数据按时间戳计数' : ''}`}>
                        Clock {inspectData.capture_clock.frequency_hz} Hz</span>}
                    {inspectData.report_job && <span className="chip blue">报告 {inspectData.report_job.slice(0, 8)}</span>}
                    {inspectData.runs.slice(0, 4).map(run => <button key={run.job_id} type="button" className="chip link-chip"
                        title={`在队列打开任务 ${run.job_id}`}
                        onClick={() => { setSelectedId(run.job_id); closeInspect(); }}>
                        {STATE_LABELS[run.state as JobState] ?? run.state ?? '任务'}</button>)}
                </div>}
                <p className="path hint" title={inspectPath}>{inspectPath}</p>
                {inspectData && <div className="inspect-time" role="group" aria-label="拍摄时间范围">
                    <span className="inspect-time-label">拍摄时间范围</span>
                    {inspectRange && Number.isFinite(inspectRange.play_start) && Number.isFinite(inspectRange.play_end) ? <>
                        起始帧 <b>{inspectRange.play_start}</b> <span className="inspect-time-arrow" aria-hidden="true">→</span> 结束帧 <b>{inspectRange.play_end}</b>
                        <span className="hint"> · 共 {inspectRange.frame_count ?? (inspectRange.play_end - inspectRange.play_start + 1)} 帧</span>
                        {inspectFps != null && <span className="hint"> · ≈ {(inspectRange.play_start / inspectFps).toFixed(2)}s → {(inspectRange.play_end / inspectFps).toFixed(2)}s @ {inspectFps} fps</span>}
                        {inspectData.capture_wall?.capture_start && <span className="hint"> · 挂钟 <b>{wallClock(inspectData.capture_wall.capture_start)}</b> → <b>{wallClock(inspectData.capture_wall.capture_end)}</b>{inspectData.capture_wall.source === 'file_mtime' ? '（按文件保存时间推算）' : ` · Live 打开 ${wallClock(inspectData.capture_wall.live_opened_utc)} @135MHz 时码`}</span>}
                    </> : inspectData.frame_rate_hz != null && inspectData.animation?.[0]?.duration_seconds != null ? <>
                        <b>{(inspectData.animation[0].start_seconds ?? 0).toFixed(2)}s</b> <span className="inspect-time-arrow" aria-hidden="true">→</span> <b>{((inspectData.animation[0].start_seconds ?? 0) + inspectData.animation[0].duration_seconds).toFixed(2)}s</b>
                        <span className="hint"> · {inspectData.animation[0].approx_frames ?? Math.round(inspectData.animation[0].duration_seconds * inspectData.frame_rate_hz)} 帧 @ {inspectData.frame_rate_hz} fps(动画片段)</span>
                    </> : <span className="hint">尚未读到时间范围:任务 load 阶段之后这里会自动显示起始/结束帧</span>}
                </div>}
                {inspectBusy && <p role="status" className="hint">正在解析（历史报告 / FBX 结构 / MCP 记录扫描）…</p>}
                {inspectError && <p className="callout" role="alert">{inspectError}</p>}
                {inspectData && <>
                    {(inspectData.frame_rate_hz != null || inspectData.animation?.[0]?.duration_seconds != null || inspectData.stats?.models != null) && <div className="inspect-facts">
                        {inspectData.frame_rate_hz != null && <span className="chip">{inspectData.frame_rate_hz} fps</span>}
                        {inspectData.animation?.[0]?.duration_seconds != null &&
                            <span className="chip" title={`动画层 ${inspectData.animation[0].name}`}>{inspectData.animation[0].duration_seconds}s
                                {inspectData.animation[0].approx_frames != null ? ` · ${inspectData.animation[0].approx_frames} 帧` : ''}</span>}
                        {inspectData.stats?.models != null && <span className="chip" title={JSON.stringify(inspectData.stats.model_kinds ?? {})}>模型 {inspectData.stats.models}</span>}
                        {inspectData.stats?.model_kinds?.Marker != null && <span className="chip" title={`已标记 ${inspectData.stats.labeled_markers ?? '?'} · 未标记 ${inspectData.stats.unlabeled_markers ?? '?'}(总 Marker 模型数)`}>marker {inspectData.stats.model_kinds.Marker}</span>}
                        {inspectData.stats?.unlabeled_markers != null && inspectData.stats.unlabeled_markers > 0 && <span className="chip amber" title="Unlabeled_Markers 分支下的 marker(无命名标签)">未标记 {inspectData.stats.unlabeled_markers}</span>}
                        {inspectData.stats?.model_kinds?.Camera != null && <span className="chip">{inspectData.stats.model_kinds.Camera} 相机</span>}
                    </div>}
                    {inspectData.notes.map((note, index) => <p key={index} className={`callout ${note.includes('人体专用') ? 'warn' : 'info'}`}>{note}</p>)}
                    <div className="roster-grid">
                        <div className="roster-list">
                            <h3>人物（{inspectData.subjects.length}）</h3>
                            {inspectData.subjects.map((item) => rosterButton('s', item))}
                            {!inspectData.subjects.length && !inspectBusy && <p className="hint">未识别到人物。</p>}
                            <h3>物体（{inspectData.objects.length}）</h3>
                            {inspectData.objects.map((item) => rosterButton('o', item))}
                            {!inspectData.objects.length && !inspectBusy && <p className="hint">未识别到物体。</p>}
                        </div>
                        <div className="roster-detail">
                            {inspectSelected ? <>
                                <div className="col-head"><h3>{inspectSelected.name}</h3><span className="grow" />
                                    {inspectSelected.sources.map((source) => <span key={source} className="chip">{SOURCE_LABELS[source] ?? source}</span>)}</div>
                                {inspectSelected.fbx_path && <p className="path hint">{inspectSelected.fbx_path}</p>}
                                {!!inspectSelected.mesh_names?.length && <p className="hint">网格:{' '}{inspectSelected.mesh_names.join('、')}</p>}
                                <div className="raw"><JsonDiagnostic value={inspectSelected.report ?? inspectSelected} /></div>
                            </> : <p className="wb-empty">点左侧条目查看具体信息（骨骼 / marker / 网格 / 报告字段）。</p>}
                        </div>
                    </div>
                </>}
            </div>
        </div>}

        {/* ── FBX 查看器为独立页面段(数据看板 | 处理台 | FBX 查看)── */}
    </main>;
}
