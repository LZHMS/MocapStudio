const BASE = '/api/processing';

export type JobState = 'QUEUED' | 'STARTING' | 'WAITING' | 'RUNNING' | 'PAUSE_REQUESTED' | 'SUCCESS' | 'NEEDS_REVIEW' | 'FAILED' | 'CANCELLED' | 'INTERRUPTED';
export type JobAction = 'step' | 'run' | 'pause' | 'cancel' | 'preview' | 'retry' | 'approve';
export type InputKind = 'mcp' | 'vdf' | 'fbx';

export interface ProcessingOptions {
    filter_markers: boolean;
    expected_subjects: string[] | null;
    expected_props: string[] | null;
    dataset_mode: boolean;
    attach_meshes: boolean;
    lightweight: boolean;
    drop_unused_labeling: boolean;
    denoise_static: boolean;
    prop_jitter_max_correction: number;
    prop_tolerance: number;
}

export interface ProcessingStage {
    id: string;
    index: number;
    name: string;
    label?: string;
    subject?: string;
    status: 'WAITING' | 'RUNNING' | 'SUCCESS' | 'FAILED' | 'SKIPPED';
    started_at?: string;
    finished_at?: string;
    elapsed_seconds?: number;
    error?: string;
    checkpoint?: string;
}

export interface ProcessingArtifact {
    id: string;
    name: string;
    role: string;
    path: string;
    size_bytes: number;
    url: string;
    stage_id?: string;
}

export interface ProcessingJob {
    id: string;
    name: string;
    input_path: string;
    input_kind: InputKind;
    mode: 'manual' | 'auto';
    state: JobState;
    created_at: string;
    updated_at: string;
    current_stage: string | null;
    waiting_stage: string | null;
    error: string | null;
    review?: { approved_at: string; note: string } | null;
    stages: ProcessingStage[];
    artifacts: ProcessingArtifact[];
    options: ProcessingOptions;
    quality_status?: string;
    worker_pid?: number;
}

export interface ProcessingConfig {
    data_roots: string[];
    output_root: string;
    default_browse_path: string;
    shogun_available: boolean;
    shogun_path: string | null;
    blender_path: string | null;
    max_workers: number;
    max_workers_limit: number;
    viewer_base: string;
    csrf_token: string;
    stages: { id: string; label: string; description: string }[];
}

export interface ProcessingSequence {
    id: string;
    name: string;
    path: string;
    kind: InputKind;
    size_bytes: number;
    modified_at: string;
    x2d_exists: boolean;
}

export interface BrowseResult {
    path: string;
    parent: string | null;
    roots: string[];
    directories: { name: string; path: string }[];
    sequences: ProcessingSequence[];
    truncated: boolean;
}

export type TreeEntry =
    | { name: string; path: string; kind: 'dir' }
    | { name: string; path: string; kind: InputKind; size_bytes: number; modified_at: string; x2d_exists: boolean; deletable?: boolean };

export interface TreeResult {
    path: string;
    parent: string | null;
    roots: string[];
    entries: TreeEntry[];
    truncated: boolean;
}

export interface FindMatch { name: string; path: string; kind: InputKind; size_bytes: number; x2d_exists: boolean }

export interface OutputEntry { name: string; path: string; kind: InputKind; size_bytes: number; modified_at: string; deletable: boolean }
export interface ProgressGroup { name: string; total: number; done: number }
export interface ProgressFamily { total: number; done: number; groups: ProgressGroup[] }
export interface FindResult { query: string; matches: FindMatch[]; scanned: number; truncated: boolean }

export interface HardwareGpu { vendor: string; name: string; driver: string | null; memory_mb: number | null }
export interface HardwareInfo {
    system: { platform: string; release: string; machine: string };
    cpu: { name: string; logical_cores: number | null };
    memory_total_bytes: number | null;
    gpus: HardwareGpu[];
    python: string;
}

export interface InspectRun { job_id: string; state: JobState | null; mode: string | null; created_at: string; updated_at: string | null }
export interface InspectItem {
    name: string; sources: string[];
    bones?: number; markers?: number; meshes?: number; nodes?: number; occurrences?: number;
    rigid?: boolean;
    mesh_names?: string[]; fbx_path?: string;
    rig_class?: string | null; classification?: string | null;
    report?: Record<string, unknown>;
}
export interface InspectPayload {
    path: string; kind: InputKind; size_bytes: number; modified_at: string;
    runs: InspectRun[]; report_job?: string; x2d_exists: boolean;
    subjects: InspectItem[]; objects: InspectItem[];
    stats?: { models?: number; model_kinds?: Record<string, number>; object_counts?: Record<string, number>;
        labeled_markers?: number; unlabeled_markers?: number;
        unlisted_branches?: { name: string; kind: string; nodes: number }[] };
    animation?: { name: string; start_seconds?: number; duration_seconds?: number; approx_frames?: number }[];
    frame_rate_hz?: number | null;
    capture_clock?: { frequency_hz?: number; timestamped?: boolean } | null;
    capture_wall?: { capture_start?: string; capture_end?: string; live_opened_utc?: string;
        seconds_from_live_open?: number[]; timecode_mhz?: number; source?: string; note?: string };
    dataset_family?: string | null;
    report?: Record<string, unknown>;
    notes: string[];
}

export interface MetricsDisk { path: string; total_bytes: number; free_bytes: number }
export interface MetricsInfo {
    cpu_percent: number | null;
    memory_used_bytes: number | null; memory_total_bytes: number | null;
    disk: MetricsDisk[]; uptime_seconds: number | null;
}

export class ProcessingApiError extends Error {
    constructor(public readonly status: number, message: string) {
        super(message);
        this.name = 'ProcessingApiError';
    }
}

async function request<T>(path: string, signal?: AbortSignal, body?: unknown, token?: string): Promise<T> {
    const response = await fetch(`${BASE}${path}`, {
        method: body === undefined ? 'GET' : 'POST',
        credentials: 'same-origin',
        cache: 'no-store',
        signal,
        headers: body === undefined ? { Accept: 'application/json' } : {
            Accept: 'application/json',
            'Content-Type': 'application/json',
            'X-Mocap-Token': token ?? '',
        },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    const text = await response.text();
    let data: unknown;
    try {
        data = text ? JSON.parse(text) : null;
    } catch {
        throw new ProcessingApiError(response.status, response.ok
            ? '服务返回了无效 JSON。'
            : `HTTP ${response.status}: ${text.slice(0, 500) || response.statusText}`);
    }
    if (!response.ok) {
        const error = data && typeof data === 'object' ? data as Record<string, unknown> : null;
        const message = error?.error ?? error?.message;
        throw new ProcessingApiError(response.status, typeof message === 'string'
            ? message : `HTTP ${response.status}: ${JSON.stringify(data)}`);
    }
    return data as T;
}

const jobPath = (id: string) => `/jobs/${encodeURIComponent(id)}`;

export const processingApi = {
    config: (signal?: AbortSignal) => request<ProcessingConfig>('/config', signal),
    hardware: (signal?: AbortSignal) => request<{ hardware: HardwareInfo }>('/hardware', signal),
    metrics: (signal?: AbortSignal) => request<{ metrics: MetricsInfo }>('/metrics', signal),
    setWorkers: (max_workers: number, token: string) => request<ProcessingConfig>('/config', undefined, { max_workers }, token),
    browse: (path: string, recursive: boolean, signal?: AbortSignal) => request<BrowseResult>(
        `/browse?${new URLSearchParams({ path, recursive: String(recursive) })}`, signal,
    ),
    tree: (path: string | null, signal?: AbortSignal) => request<TreeResult>(
        `/tree${path ? `?${new URLSearchParams({ path })}` : ''}`, signal,
    ),
    find: (q: string, signal?: AbortSignal) => request<FindResult>(`/find?${new URLSearchParams({ q })}`, signal),
    outputs: (limit = 24, signal?: AbortSignal) => request<{ outputs: OutputEntry[] }>(`/outputs?${new URLSearchParams({ limit: String(limit) })}`, signal),
    datasetProgress: (signal?: AbortSignal) => request<{ families: Record<string, ProgressFamily> }>('/dataset-progress', signal),
    inspect: (path: string, signal?: AbortSignal) => request<InspectPayload>(`/inspect?${new URLSearchParams({ path })}`, signal),
    jobs: (signal?: AbortSignal) => request<{ jobs: ProcessingJob[] }>('/jobs', signal),
    job: (id: string, signal?: AbortSignal) => request<{ job: ProcessingJob }>(jobPath(id), signal),
    createJobs: (paths: string[], mode: 'manual' | 'auto', options: ProcessingOptions, token: string) =>
        request<{ jobs: ProcessingJob[] }>('/jobs', undefined, { paths, mode, options }, token),
    action: (id: string, action: JobAction, token: string, note?: string) =>
        request<{ job: ProcessingJob }>(`${jobPath(id)}/actions`, undefined, note === undefined ? { action } : { action, note }, token),
    deleteJob: (id: string, token: string) => request<{ removed: string }>(`${jobPath(id)}/delete`, undefined, {}, token),
    clearArchive: (id: string, token: string) => request<{ removed: string[] }>(`${jobPath(id)}/clear-archive`, undefined, {}, token),
    deleteOutput: (path: string, token: string) => request<{ deleted: string }>('/delete-output', undefined, { path }, token),
    clearFinished: (token: string) => request<{ removed: string[]; count: number }>('/jobs/clear-finished', undefined, {}, token),
    logs: (id: string, offset: number, signal?: AbortSignal) => request<{ text: string; next_offset: number }>(
        `${jobPath(id)}/logs?offset=${offset}`, signal,
    ),
    report: (id: string, signal?: AbortSignal) => request<{ report: Record<string, unknown> | null }>(
        `${jobPath(id)}/report`, signal,
    ),
};
