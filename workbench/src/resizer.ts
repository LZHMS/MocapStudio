/**
 * 列宽拖拽辅助模块(纯 DOM / 事件,零依赖,自包含)。
 *
 * 接入方式(Processing.tsx,拿到 .cols 的 ref 之后):
 *   useEffect(() => {
 *       const host = colsRef.current;
 *       return host ? attachColumnResizers(host) : undefined;
 *   }, []);
 *
 * 手柄以 absolutely-positioned div.col-resizer 的形式注入到第 1 / 第 2 列内部右缘,
 * 不影响 grid 轨道;processing.css 中的 .col-resizer / .col-resize-host / .is-resizing
 * 规则负责外观与窄屏隐藏。仅宽屏(≥1181px)启用,跨断点自动加卸手柄。
 */

const STORAGE_KEY = 'mocap-col-widths';
const ENABLE_QUERY = '(min-width: 1181px)';

const COLUMN_SELECTOR = ':scope > .col';
const HANDLE_SELECTOR = ':scope > .col-resizer';
const HOST_CLASS = 'col-resize-host';
const HANDLE_CLASS = 'col-resizer';
const DRAG_CLASS = 'is-dragging';
const RESIZING_CLASS = 'is-resizing';

const DEFAULT_FIRST = 330;
const DEFAULT_SECOND = 280;
const MIN_FIRST = 200;
const MIN_SECOND = 180;
const MIN_REMAINDER = 320;
const FALLBACK_GAP = 12;
const KEY_STEP = 8;
const KEY_STEP_FAST = 16;

export interface ColumnWidths {
    first: number;
    second: number;
}

/** 默认列宽(与 .cols 的 grid-template-columns 保持一致)。 */
export const DEFAULT_COLUMN_WIDTHS: Readonly<ColumnWidths> = { first: DEFAULT_FIRST, second: DEFAULT_SECOND };

/** 清除持久化的列宽(下次 attach 回到 CSS 默认)。 */
export function clearStoredColumnWidths(): void {
    try {
        window.localStorage.removeItem(STORAGE_KEY);
    } catch {
        /* localStorage 不可用:忽略 */
    }
}

const roundNumber = (value: unknown): number | null =>
    typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : null;

const clamp = (value: number, min: number, max: number): number => {
    if (!Number.isFinite(value)) return min;
    const upper = Math.max(min, max);
    return Math.min(upper, Math.max(min, Math.round(value)));
};

function readStoredWidths(): ColumnWidths | null {
    try {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (!raw) return null;
        const parsed: unknown = JSON.parse(raw);
        if (typeof parsed !== 'object' || parsed === null) return null;
        const record = parsed as Record<string, unknown>;
        const first = roundNumber(record.first);
        const second = roundNumber(record.second);
        if (first === null || second === null) return null;
        return { first, second };
    } catch {
        return null;
    }
}

function writeStoredWidths(widths: ColumnWidths): void {
    try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(widths));
    } catch {
        /* 隐私模式 / 配额超限:持久化失败不影响拖拽 */
    }
}

/**
 * 给 host(.cols)的第 1 与第 2 列右缘各挂一个拖拽手柄。
 * 幂等:同一列已有手柄则复用,不会重复插入。
 * 返回卸载函数:移除手柄与所有监听,并撤销内联 grid-template-columns。
 */
export function attachColumnResizers(host: HTMLElement): () => void {
    const media: MediaQueryList | null =
        typeof window.matchMedia === 'function' ? window.matchMedia(ENABLE_QUERY) : null;

    const state: ColumnWidths = readStoredWidths() ?? { first: DEFAULT_FIRST, second: DEFAULT_SECOND };

    const handleCleanups: Array<() => void> = [];
    const globalCleanups: Array<() => void> = [];
    const handles: HTMLElement[] = [];

    let inlineApplied = false;
    let disposed = false;

    /** 前两列(grid 轨道 1 / 2)。 */
    const columns = (): HTMLElement[] => {
        const found = host.querySelectorAll<HTMLElement>(COLUMN_SELECTOR);
        const list: HTMLElement[] = [];
        for (let i = 0; i < found.length && list.length < 2; i += 1) list.push(found[i]);
        return list;
    };

    const gapPx = (): number => {
        const raw = Number.parseFloat(window.getComputedStyle(host).columnGap);
        return Number.isFinite(raw) && raw >= 0 ? raw : FALLBACK_GAP;
    };

    const minFor = (index: number): number => (index === 0 ? MIN_FIRST : MIN_SECOND);

    /** 被拖列的上限:容器内容宽 - 两个列间距 - 其余列的最小宽。 */
    const maxFor = (index: number): number => {
        const others = index === 0 ? MIN_SECOND + MIN_REMAINDER : state.first + MIN_REMAINDER;
        return host.clientWidth - gapPx() * 2 - others;
    };

    /** 实测当前列宽,保证从任意来源(CSS 默认 / 内联值 / 持久值)接着拖。 */
    const measure = (): ColumnWidths | null => {
        const found = columns();
        if (found.length < 2) return null;
        return {
            first: Math.round(found[0].getBoundingClientRect().width),
            second: Math.round(found[1].getBoundingClientRect().width),
        };
    };

    const paint = (): void => {
        host.style.gridTemplateColumns = `${state.first}px ${state.second}px minmax(0, 1fr)`;
        inlineApplied = true;
        for (const handle of handles) {
            const index = handle.dataset.index === '1' ? 1 : 0;
            handle.setAttribute('aria-valuenow', String(index === 0 ? state.first : state.second));
        }
    };

    const commit = (): void => writeStoredWidths(state);

    const bindHandle = (handle: HTMLElement, index: number): void => {
        let pointerId: number | null = null;
        let startX = 0;
        let base: ColumnWidths = { ...state };
        let bodyCursor = '';

        const listen = <T extends keyof HTMLElementEventMap>(
            type: T,
            listener: (event: HTMLElementEventMap[T]) => void,
        ): void => {
            handle.addEventListener(type, listener);
            handleCleanups.push(() => handle.removeEventListener(type, listener));
        };

        const setValue = (value: number): void => {
            if (index === 0) state.first = value;
            else state.second = value;
        };

        const currentValue = (): number => (index === 0 ? state.first : state.second);

        const release = (): void => {
            if (pointerId === null) return;
            pointerId = null;
            handle.classList.remove(DRAG_CLASS);
            host.classList.remove(RESIZING_CLASS);
            window.document.body.style.cursor = bodyCursor;
            commit();
        };

        listen('pointerdown', (event) => {
            if (event.pointerType === 'mouse' && event.button !== 0) return;
            const measured = measure();
            if (!measured) return;
            event.preventDefault();
            base = measured;
            state.first = measured.first;
            state.second = measured.second;
            startX = event.clientX;
            pointerId = event.pointerId;
            bodyCursor = window.document.body.style.cursor;
            window.document.body.style.cursor = 'col-resize';
            handle.classList.add(DRAG_CLASS);
            host.classList.add(RESIZING_CLASS);
            try {
                handle.setPointerCapture(event.pointerId);
            } catch {
                /* 指针捕获不可用时退回普通事件流 */
            }
        });

        listen('pointermove', (event) => {
            if (pointerId === null || event.pointerId !== pointerId) return;
            const from = index === 0 ? base.first : base.second;
            const next = clamp(from + (event.clientX - startX), minFor(index), maxFor(index));
            state.first = index === 0 ? next : base.first;
            state.second = index === 0 ? base.second : next;
            paint();
        });

        listen('pointerup', release);
        listen('pointercancel', release);
        listen('lostpointercapture', release);

        listen('dblclick', () => {
            setValue(index === 0 ? DEFAULT_FIRST : DEFAULT_SECOND);
            paint();
            commit();
        });

        listen('keydown', (event) => {
            const step = event.shiftKey ? KEY_STEP_FAST : KEY_STEP;
            let next: number;
            if (event.key === 'ArrowLeft') next = currentValue() - step;
            else if (event.key === 'ArrowRight') next = currentValue() + step;
            else if (event.key === 'Home' || event.key === 'Escape') next = index === 0 ? DEFAULT_FIRST : DEFAULT_SECOND;
            else return;
            event.preventDefault();
            setValue(clamp(next, minFor(index), maxFor(index)));
            paint();
            commit();
        });
    };

    const createHandles = (): void => {
        const found = columns();
        if (found.length < 2) return;
        found.forEach((column, index) => {
            column.classList.add(HOST_CLASS);
            let handle = column.querySelector<HTMLElement>(HANDLE_SELECTOR);
            if (!handle) {
                handle = window.document.createElement('div');
                handle.className = HANDLE_CLASS;
                column.appendChild(handle);
            }
            handle.dataset.index = String(index);
            handle.setAttribute('role', 'separator');
            handle.setAttribute('aria-orientation', 'vertical');
            handle.tabIndex = 0;
            handle.setAttribute(
                'aria-label',
                index === 0
                    ? '拖拽调整「数据浏览器」列宽(方向键微调,双击恢复默认)'
                    : '拖拽调整「处理队列」列宽(方向键微调,双击恢复默认)',
            );
            handle.setAttribute('aria-valuenow', String(index === 0 ? state.first : state.second));
            handles.push(handle);
            bindHandle(handle, index);
        });
    };

    const deactivate = (): void => {
        for (const off of handleCleanups) off();
        handleCleanups.length = 0;
        for (const handle of handles) handle.remove();
        handles.length = 0;
        for (const column of Array.from(host.querySelectorAll<HTMLElement>(COLUMN_SELECTOR))) {
            column.classList.remove(HOST_CLASS);
        }
        host.classList.remove(RESIZING_CLASS);
        if (inlineApplied) {
            host.style.removeProperty('grid-template-columns');
            inlineApplied = false;
        }
    };

    const sync = (): void => {
        if (disposed) return;
        const enabled = media === null || media.matches;
        if (enabled && handles.length === 0) {
            createHandles();
            const stored = readStoredWidths();
            // 有持久值且 host 上没有内联列宽时才套用,避免覆盖调用方的显式设置。
            if (stored !== null && host.style.gridTemplateColumns === '') {
                state.first = Math.max(MIN_FIRST, stored.first);
                state.second = Math.max(MIN_SECOND, stored.second);
                paint();
            }
        } else if (!enabled && handles.length > 0) {
            // 窄屏:移除手柄并撤销内联列宽,让 CSS 媒体查询的堆叠布局接管。
            deactivate();
        }
    };

    sync();

    if (media !== null) {
        const onMediaChange = (): void => sync();
        media.addEventListener('change', onMediaChange);
        globalCleanups.push(() => media.removeEventListener('change', onMediaChange));
    }

    return () => {
        if (disposed) return;
        disposed = true;
        for (const off of globalCleanups) off();
        globalCleanups.length = 0;
        deactivate();
    };
}
