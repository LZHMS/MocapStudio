import { useEffect, useMemo, useRef, useState } from 'react';
import { processingApi, type HardwareInfo, type MetricsInfo, type ProcessingConfig, type ProgressFamily, type ProcessingJob } from './processingApi';
import './home.css';

export interface HomeCounts { QUEUED: number; RUNNING: number; WAITING: number; NEEDS_REVIEW: number; SUCCESS: number; PROBLEM: number }
export interface HomePanelProps {
    config: ProcessingConfig | null;
    jobs: ProcessingJob[];
    counts: HomeCounts;
    onOpenJob: (jobId: string) => void;
    onGoWork: () => void;
    onInspectFile: (path: string) => void;
}

const STATE_LABELS: Record<string, string> = {
    QUEUED: '排队中', STARTING: '启动中', WAITING: '等待操作', RUNNING: '执行中',
    PAUSE_REQUESTED: '等待安全暂停', SUCCESS: '完成', NEEDS_REVIEW: '需人工复核',
    FAILED: '失败', CANCELLED: '已取消', INTERRUPTED: '已中断',
};
const TERMINAL: Partial<Record<string, true>> = { SUCCESS: true, NEEDS_REVIEW: true, FAILED: true, CANCELLED: true, INTERRUPTED: true };

function statusTone(state: string): string {
    if (state === 'SUCCESS') return 'green';
    if (state === 'FAILED' || state === 'INTERRUPTED') return 'red';
    if (state === 'NEEDS_REVIEW' || state === 'PAUSE_REQUESTED') return 'amber';
    if (state === 'RUNNING' || state === 'STARTING') return 'blue';
    return '';
}

function gbytes(bytes: number): string {
    if (bytes >= 1024 ** 4) return `${(bytes / 1024 ** 4).toFixed(2)} TB`;
    return `${(bytes / 1024 ** 3).toFixed(0)} GB`;
}

function fmtSize(bytes: number): string {
    if (bytes >= 1024 ** 3) return gbytes(bytes);
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function fmtUptime(seconds: number): string {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return days ? `${days}天 ${hours}h ${String(minutes).padStart(2, '0')}m` : `${hours}h ${String(minutes).padStart(2, '0')}m`;
}

function shortTime(value?: string): string {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.valueOf())) return value;
    return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}

const JOB_TILES: { key: keyof HomeCounts; label: string; tone: string }[] = [
    { key: 'QUEUED', label: '排队', tone: '' },
    { key: 'RUNNING', label: '执行中', tone: 'blue' },
    { key: 'WAITING', label: '等待操作 / 暂停', tone: 'blue' },
    { key: 'NEEDS_REVIEW', label: '需人工复核', tone: 'amber' },
    { key: 'SUCCESS', label: '完成', tone: 'green' },
    { key: 'PROBLEM', label: '失败 / 中断 / 取消', tone: 'red' },
];

type SortKey = 'created_at' | 'name' | 'state';

export default function HomePanel({ config, jobs, counts, onOpenJob, onGoWork, onInspectFile }: HomePanelProps) {
    const [metrics, setMetrics] = useState<MetricsInfo | null>(null);
    const [hardware, setHardware] = useState<HardwareInfo | null>(null);
    const [progress, setProgress] = useState<Record<string, ProgressFamily> | null>(null);
    const [progressOpen, setProgressOpen] = useState<Record<string, boolean>>({});
    const [cpuHistory, setCpuHistory] = useState<number[]>([]);
    const historyRef = useRef<number[]>([]);
    const [sortKey, setSortKey] = useState<SortKey>('created_at');
    const [sortDir, setSortDir] = useState<1 | -1>(-1);

    useEffect(() => {
        let stopped = false;
        async function poll() {
            try {
                const result = await processingApi.metrics();
                if (stopped) return;
                setMetrics(result.metrics);
                historyRef.current = [...historyRef.current.slice(-23), result.metrics.cpu_percent ?? 0];
                setCpuHistory(historyRef.current);
            } catch { /* 下一轮再试 */ }
        }
        void poll();
        processingApi.hardware().then((result) => { if (!stopped) setHardware(result.hardware); }).catch(() => { /* 环境卡显示占位 */ });
        const timer = window.setInterval(() => void poll(), 2500);
        return () => { stopped = true; window.clearInterval(timer); };
    }, []);

    useEffect(() => {
        let stopped = false;
        const loadProgress = () => processingApi.datasetProgress().then((result) => { if (!stopped) setProgress(result.families); }).catch(() => { /* 下一轮再试 */ });
        void loadProgress();
        const timer = window.setInterval(() => void loadProgress(), 30000);
        return () => { stopped = true; window.clearInterval(timer); };
    }, []);

    const health = Math.max(0, 100 - counts.PROBLEM * 15 - counts.NEEDS_REVIEW * 5);
    const healthLabel = counts.PROBLEM ? '需处理' : counts.NEEDS_REVIEW ? '需注意' : '很好';
    const memPercent = metrics?.memory_used_bytes != null && metrics.memory_total_bytes
        ? Math.round(100 * metrics.memory_used_bytes / metrics.memory_total_bytes) : null;
    const cpuPercent = metrics?.cpu_percent ?? null;
    const cpuTone = cpuPercent == null ? 'blue' : cpuPercent >= 85 ? 'red' : cpuPercent >= 60 ? 'amber' : 'blue';
    const memTone = memPercent == null ? 'blue' : memPercent >= 85 ? 'red' : memPercent >= 70 ? 'amber' : 'blue';
    const disk = metrics?.disk?.[0] ?? null;
    const diskPercent = disk ? Math.round(100 * (1 - disk.free_bytes / disk.total_bytes)) : null;
    const diskTone = diskPercent == null ? 'blue' : diskPercent >= 90 ? 'red' : diskPercent >= 80 ? 'amber' : 'blue';
    const driveName = disk ? disk.path.split(/[\\/]/)[0] : '数据盘';
    const finishedStages = (job: ProcessingJob) => job.stages.filter((stage) => ['SUCCESS', 'FAILED', 'SKIPPED'].includes(stage.status)).length;

    const rows = useMemo(() => [...jobs].sort((a, b) => {
        if (sortKey === 'name') return sortDir * a.name.localeCompare(b.name, 'zh-CN', { numeric: true });
        if (sortKey === 'state') return sortDir * (a.state.localeCompare(b.state) || b.created_at.localeCompare(a.created_at));
        return sortDir * a.created_at.localeCompare(b.created_at);
    }), [jobs, sortKey, sortDir]);
    const sortMark = (key: SortKey) => (sortKey === key ? (sortDir === 1 ? ' ↑' : ' ↓') : '');
    const toggleSort = (key: SortKey) => {
        if (key === sortKey) setSortDir((previous) => (previous === 1 ? -1 : 1));
        else { setSortKey(key); setSortDir(key === 'name' ? 1 : -1); }
    };

    return <div className="home dash">
        {!config && <div className="callout" role="alert">未连接处理服务;请确认 Mocap Studio 已启动。</div>}
        <div className="dash-grid" aria-label="机器状态">
            <div className={`dash-card${counts.PROBLEM ? ' red' : counts.NEEDS_REVIEW ? ' amber' : ' green'}`}>
                <div className="dash-head"><span className="dash-name"><i className={`ico-dot ${counts.PROBLEM ? "red" : counts.NEEDS_REVIEW ? "amber" : "green"}`} />健康度</span>
                    <span className={`dash-pill ${counts.PROBLEM ? 'red' : counts.NEEDS_REVIEW ? 'amber' : 'green'}`}>{healthLabel}</span></div>
                <div className="dash-value"><b>{health}</b><span className="dash-unit">/100</span></div>
                <p className="dash-sub">{counts.PROBLEM ? `${counts.PROBLEM} 个异常任务待处理` : counts.NEEDS_REVIEW ? `${counts.NEEDS_REVIEW} 个任务需人工复核` : '各项指标正常'}</p>
                <p className="dash-foot">数据根 {config?.data_roots.length ?? 0} 个 · 回环服务 · 原始输入不覆盖</p>
            </div>
            <div className={`dash-card ${cpuTone}`}>
                <div className="dash-head"><span className="dash-name"><i className="ico-dot blue" />CPU</span>
                    <span className="dash-pill">{hardware?.cpu.logical_cores ?? '?'} 核</span></div>
                <div className="dash-value"><b>{metrics?.cpu_percent == null ? '…' : Math.round(metrics.cpu_percent)}</b><span className="dash-unit">%</span></div>
                <div className="dash-spark" aria-hidden="true">{cpuHistory.map((value, index) =>
                    <i key={index} style={{ height: `${Math.max(6, Math.min(100, value))}%` }} />)}</div>
                <p className="dash-sub">{hardware?.cpu.name || '正在读取本机信息…'}</p>
            </div>
            <div className="dash-card">
                <div className="dash-head"><span className="dash-name"><i className="ico-dot muted" />GPU</span>
                    <span className="dash-pill">{hardware?.gpus.length ? hardware.gpus[0].vendor : '—'}</span></div>
                <div className="dash-value"><b>{hardware?.gpus.length ?? '…'}</b><span className="dash-unit">块</span></div>
                <p className="dash-sub">{hardware?.gpus[0]?.name || '未检测到 GPU'}</p>
                <p className="dash-foot">{hardware?.gpus[0]?.memory_mb ? `显存 ${Math.round(hardware.gpus[0].memory_mb / 1024)} GB` : ''}</p>
            </div>
            <div className={`dash-card ${memTone}`}>
                <div className="dash-head"><span className="dash-name"><i className="ico-dot blue" />内存</span>
                    <span className={`dash-pill ${memPercent != null && memPercent > 85 ? 'red' : memPercent != null && memPercent > 70 ? 'amber' : ''}`}>压力 {memPercent == null ? '…' : `${memPercent}%`}</span></div>
                <div className="dash-value"><b>{memPercent == null ? '…' : memPercent}</b><span className="dash-unit">%</span></div>
                <div className="dash-meter"><span style={{ width: `${memPercent ?? 0}%` }} /></div>
                <p className="dash-sub">{metrics?.memory_used_bytes != null && metrics.memory_total_bytes
                    ? `${gbytes(metrics.memory_used_bytes)} / ${gbytes(metrics.memory_total_bytes)}` : '正在读取…'}</p>
            </div>
            <div className={`dash-card ${diskTone}`}>
                <div className="dash-head"><span className="dash-name"><i className="ico-dot amber" />磁盘</span>
                    <span className={`dash-pill ${diskPercent != null && diskPercent > 90 ? 'red' : diskPercent != null && diskPercent > 80 ? 'amber' : ''}`}>{disk ? gbytes(disk.total_bytes) : '…'}</span></div>
                <div className="dash-value"><b>{disk ? gbytes(disk.free_bytes) : '…'}</b><span className="dash-unit">可用</span></div>
                <div className="dash-meter"><span style={{ width: `${diskPercent ?? 0}%` }} /></div>
                <p className="dash-sub">{disk ? `${driveName} 已用 ${diskPercent}%(数据根所在盘)` : '正在读取…'}</p>
            </div>
            <div className="dash-card">
                <div className="dash-head"><span className="dash-name"><i className="ico-dot green" />开机时长</span>
                    <span className="dash-pill">本机</span></div>
                <div className="dash-value"><b>{metrics?.uptime_seconds != null ? fmtUptime(metrics.uptime_seconds) : '…'}</b></div>
                <p className="dash-sub">Shogun {config ? (config.shogun_available ? '✓' : '✗') : '…'} · Blender {config ? (config.blender_path ? '✓' : '✗') : '…'} · 并发上限 {config?.max_workers ?? '…'}</p>
            </div>
        </div>

        <div className="dash-grid jobs" aria-label="处理状态">
            {JOB_TILES.map((tile) => <button key={tile.key} type="button" className={`dash-card job${counts[tile.key] ? '' : ' dim'}${tile.tone ? ` ${tile.tone}` : ''}`}
                onClick={onGoWork} title="进入处理台">
                <div className="dash-head"><span className="dash-name">{tile.label}</span></div>
                <div className="dash-value"><b>{counts[tile.key]}</b></div>
            </button>)}
        </div>

        <div className="dash-duo">
        <section className="panel dash-table-panel">
            <div className="col-head"><h2>最近运行 <span className="count">共 {jobs.length} 条</span></h2><span className="grow" />
                <span className="hint">点表头排序 · 点行进入处理台详情</span></div>
            {!rows.length && <p className="wb-empty">尚无任务。在「处理台」选序列后创建运行;第一幕 0001-parse 会自动给出名单。</p>}
            {!!rows.length && <div className="dash-table-scroll">
                <table className="dash-table">
                    <thead><tr>
                        <th role="button" tabIndex={0} onClick={() => toggleSort('name')} onKeyDown={(event) => event.key === 'Enter' && toggleSort('name')}>名称{sortMark('name')}</th>
                        <th role="button" tabIndex={0} onClick={() => toggleSort('state')} onKeyDown={(event) => event.key === 'Enter' && toggleSort('state')}>状态{sortMark('state')}</th>
                        <th>方式</th>
                        <th>PID</th>
                        <th>阶段</th>
                        <th role="button" tabIndex={0} onClick={() => toggleSort('created_at')} onKeyDown={(event) => event.key === 'Enter' && toggleSort('created_at')}>时间{sortMark('created_at')}</th>
                    </tr></thead>
                    <tbody>{rows.map((entry) => {
                        const done = finishedStages(entry);
                        const total = entry.stages.length;
                        return <tr key={entry.id} onClick={() => onOpenJob(entry.id)}>
                            <td className="dt-name">{entry.name}</td>
                            <td><span className={`chip ${statusTone(entry.state)}`}>{STATE_LABELS[entry.state] ?? entry.state}</span></td>
                            <td className="dt-meta">{entry.mode === 'manual' ? '逐步' : '自动'} · {entry.input_kind.toUpperCase()}</td>
                            <td className="dt-meta mono">{entry.worker_pid && !TERMINAL[entry.state] ? entry.worker_pid : '—'}</td>
                            <td className="dt-progress"><span className="dt-bar"><i style={{ width: `${total ? Math.round(100 * done / total) : 0}%` }} /></span>
                                <span className="dt-meta">{total ? `${done}/${total}` : '—'}</span></td>
                            <td className="dt-meta">{shortTime(entry.created_at)}</td>
                        </tr>;
                    })}</tbody>
                </table>
            </div>}
        </section>

        <section className="panel dash-outputs" aria-label="处理数据">
            <div className="col-head"><h2>处理数据
                <span className="count">{progress
                    ? `${Object.values(progress).reduce((sum, f) => sum + f.done, 0)}/${Object.values(progress).reduce((sum, f) => sum + f.total, 0)} 条序列已产出`
                    : '…'}</span></h2><span className="grow" />
                <span className="hint" title="按数据集族(BODY / MP / HOI)分组;每个演员或组的进度 = 已在 03_exports 产出可用 FBX 的 take 数 / 原始序列数">原始 01_data_raw · 产出 03_exports</span></div>
            {progress === null && <p role="status" className="wb-empty">统计原始序列与产出…</p>}
            {progress && <div className="progress-scroll">{Object.entries(progress).map(([family, info]) => <div key={family} className="progress-family"
                style={{ flexGrow: progressOpen[family] !== false ? 1 : 0, flexBasis: 0 }}>
                <button type="button" className={`pf-head${progressOpen[family] === false ? ' closed' : ''}`}
                    title="点击折叠/展开该族分组"
                    onClick={() => setProgressOpen((previous) => ({ ...previous, [family]: previous[family] === false }))}>
                    <span className="tw" aria-hidden="true" />
                    <strong>{family}</strong>
                    <span className="pf-bar"><i style={{ width: `${info.total ? Math.round(100 * info.done / info.total) : 0}%` }} /></span>
                    <span className="pf-count mono">{info.done}/{info.total}</span>
                </button>
                {progressOpen[family] !== false && <div className="pf-groups">{info.groups.map((group) => <span key={group.name} className={`pf-group${group.total && group.done >= group.total ? ' full' : group.done ? ' part' : ' todo'}`}
                    style={group.total && group.done >= group.total ? { background: 'rgba(111, 186, 44, .42)', borderColor: 'rgba(111, 186, 44, .55)' }
                        : group.done ? { background: `linear-gradient(90deg, rgba(25, 200, 185, .5) ${Math.round(100 * group.done / group.total)}%, var(--surface) ${Math.round(100 * group.done / group.total)}%)`, borderColor: 'rgba(25, 200, 185, .5)' }
                            : undefined}
                    title={`${group.name}: ${group.done}/${group.total} 条已产出`}>
                    <strong>{group.name}</strong>
                    <span className="pf-count mono small">{group.done}/{group.total}</span>
                </span>)}</div>}
            </div>)}</div>}
        </section>
        </div>
    </div>;
}
