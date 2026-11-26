import { useState } from 'react';
import { processingApi, type SmplArtifact, type SmplJob } from './processingApi';

const ACTIVE_STATES = ['QUEUED', 'STARTING', 'RUNNING'];
const STATE_LABELS: Record<string, string> = {
    QUEUED: '排队中', STARTING: '启动中', RUNNING: '执行中',
    SUCCESS: '完成', FAILED: '失败', CANCELLED: '已取消', INTERRUPTED: '已中断',
};
const STATE_TONES: Record<string, string> = {
    SUCCESS: 'green', FAILED: 'red', CANCELLED: 'amber', INTERRUPTED: 'amber',
    RUNNING: 'blue', STARTING: 'blue', QUEUED: 'faint',
};

function fileSize(bytes: number): string {
    if (!Number.isFinite(bytes)) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
    return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

export function SmplQueueSection({ jobs, refresh, onOpenArtifact, token, errorText }: {
    jobs: SmplJob[] | null;
    refresh: () => void;
    onOpenArtifact: (artifact: SmplArtifact) => void;
    token?: string;
    errorText: (error: unknown) => string;
}) {
    const [open, setOpen] = useState(true);
    const [expanded, setExpanded] = useState('');
    const [notice, setNotice] = useState('');
    const [error, setError] = useState('');
    const running = jobs?.filter((entry) => ACTIVE_STATES.includes(entry.state)).length ?? 0;
    const finished = jobs?.filter((entry) => !ACTIVE_STATES.includes(entry.state)).length ?? 0;

    async function act(fn: () => Promise<unknown>) {
        setError('');
        try { await fn(); refresh(); } catch (err) { setError(errorText(err)); }
    }

    return <div className={`queue-group${open ? '' : ' closed'} smpl-group`}>
        <button type="button" className="queue-group-head" aria-expanded={open} onClick={() => setOpen((previous) => !previous)}>
            <span className="tw" aria-hidden="true" />
            <strong>SMPL 拟合队列</strong>
            <span className="count-chip">{jobs?.length ?? 0}</span>
            {running > 0 && <span className="hint smpl-running">拟合中 {running}</span>}
            {finished > 0 && token && <button className="btn ghost smpl-clear" type="button"
                onClick={(event) => { event.stopPropagation(); void act(() => processingApi.clearSmplFinished(token)); }}>清空已结束 ({finished})</button>}
        </button>
        {open && <div className="queue-group-body smpl-body">
            {!jobs?.length && <p className="wb-empty">尚无拟合任务。在「FBX 查看」打开 03_exports 的序列后点「SMPL 拟合」，对其中每位演员分别拟合 SMPL-H。</p>}
            {jobs?.map((job) => {
                const done = job.stages.filter((stage) => stage.status === 'SUCCESS').length;
                const isExpanded = expanded === job.id;
                const isActive = ACTIVE_STATES.includes(job.state);
                return <div key={job.id} className="smpl-job">
                    <div className="smpl-row" role="button" tabIndex={0} onClick={() => setExpanded(isExpanded ? '' : job.id)}
                        onKeyDown={(event) => { if (event.key === 'Enter') setExpanded(isExpanded ? '' : job.id); }}>
                        <strong className="qi-title">{job.name.replace(/\.fbx$/i, '')}</strong>
                        <span className="qi-id mono" title={`拟合 id:${job.id}`}>#{job.id.slice(0, 8)}</span>
                        <span className={`qi-state ${STATE_TONES[job.state] ?? 'faint'}`}>{STATE_LABELS[job.state] ?? job.state}</span>
                        <span className="hint">{job.current_stage || `${done}/${job.stages.length} 阶段`}</span>
                        <span className="grow" />
                        {isActive && token && <button className="btn ghost" type="button"
                            onClick={(event) => { event.stopPropagation(); void act(() => processingApi.smplAction(job.id, 'cancel', token)); }}>取消</button>}
                        {!isActive && token && <button className="btn ghost" type="button"
                            onClick={(event) => { event.stopPropagation(); void act(() => processingApi.smplAction(job.id, 'retry', token)); }}>重试</button>}
                        {!isActive && token && <button className="btn ghost" type="button"
                            onClick={(event) => { event.stopPropagation(); void act(() => processingApi.deleteSmplJob(job.id, token)); }}>删除</button>}
                        <span className="hint">{isExpanded ? '▲' : '▼'}</span>
                    </div>
                    {isExpanded && <div className="smpl-detail">
                        <p className="hint smpl-source" title={job.input_path}>源:{job.input_path}</p>
                        {notice && <p className="hint">{notice}</p>}
                        {!!job.stages.length && <ul className="smpl-stages">
                            {job.stages.map((stage) => <li key={stage.id}>
                                <span className={`smpl-stage-${stage.status.toLowerCase()}`}>{stage.name}</span>
                                <span className="hint"> · {stage.status}{stage.error ? ` · ${stage.error}` : ''}</span>
                            </li>)}
                        </ul>}
                        {job.error && <p className="callout" role="alert">{job.error}</p>}
                        {!!job.artifacts.length && <div className="smpl-artifacts">
                            {job.artifacts.map((artifact) => <span key={artifact.id} className="smpl-artifact">
                                {artifact.role === 'smpl_fit_fbx'
                                    ? <button className="btn outline" type="button" onClick={() => onOpenArtifact(artifact)}>查看 {artifact.name}</button>
                                    : <a className="btn ghost" href={`/api/processing/output-file?path=${encodeURIComponent(artifact.path)}`}>{artifact.name} · {fileSize(artifact.size_bytes)}</a>}
                            </span>)}
                        </div>}
                    </div>}
                </div>;
            })}
            {error && <p className="callout" role="alert">{error}</p>}
        </div>}
    </div>;
}
