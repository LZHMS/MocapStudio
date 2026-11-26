import './advanced.css';

// 与 Processing.tsx 内联块保持一致的常量拷贝（value/label/detail 一字不差）。
// 简短标签 + 悬停完整说明：界面文字尽量短，语义不丢失。
const OPTION_LABELS = [
    // 按流水线实际执行顺序排列(先发生的在前)。
    { value: 'dataset_mode', label: '数据集模式', detail: '数据集模式 / 主题与预期名单审计（适用于 MCP、VDF）' },
    { value: 'filter_markers', label: '标记点过滤', detail: '过滤标记点数据（适用于 MCP / VDF）' },
    { value: 'denoise_static', label: '静止段去抖', detail: '解算后对基本静止的肢段（手指等）在静止窗口内做零相位低通，抑制非物理高频抖动；运动段不动' },
    { value: 'attach_meshes', label: '挂载物体网格', detail: '挂载现有物体网格（需要 Blender 和对应资产）' },
    { value: 'lightweight', label: '轻量显示 FBX', detail: '生成轻量显示 FBX（无损显示裁剪；不修复源文件物理质量）' },
    { value: 'drop_unused_labeling', label: '移除未用 Labeling', detail: '轻量化时移除未使用的 Labeling；保留必需锚点' },
];

export interface AdvancedOptionsProps {
    flags: string[];
    onFlags(next: string[]): void;
    jitter: string;
    onJitter(v: string): void;
    tolerance: string;
    onTolerance(v: string): void;
}

// 自包含受控组件：只负责展示与回调，状态仍由调用方（Processing.tsx）持有。
// 明确不再包含「预期人体名单 / 预期物体名单」输入——名单由 0001-parse 自动记录。
export default function AdvancedOptions({ flags, onFlags, jitter, onJitter, tolerance, onTolerance }: AdvancedOptionsProps) {
    function toggle(value: string, on: boolean): void {
        onFlags(on ? [...flags, value] : flags.filter((entry) => entry !== value));
    }

    // 与 Processing.tsx 的 optionsValid 判定等价，仅用于就地提示；提交校验仍由调用方负责。
    const jitterValid = jitter.trim() !== '' && Number.isFinite(Number(jitter)) && Number(jitter) >= 0;
    const toleranceValid = tolerance.trim() !== '' && Number.isFinite(Number(tolerance)) && Number(tolerance) > 0;

    return <div className="adv">
        <div className="adv-options">{OPTION_LABELS.map((option) => <label className="adv-check" key={option.value} title={option.detail}>
            <input type="checkbox" checked={flags.includes(option.value)}
                onChange={(event) => toggle(option.value, event.target.checked)} />
            <span>{option.label}</span>
        </label>)}</div>
        <div className="adv-sec-title">数值参数</div>
        <div className="adv-numbers">
            <label className="adv-field"><span>抖动修正 mm</span>
                <input className="input" type="number" min="0" step="any" value={jitter}
                    title="物体抖动最大修正（mm，0 表示不修正）"
                    onChange={(event) => onJitter(event.target.value)} /></label>
            <label className="adv-field"><span>刚体容差 mm</span>
                <input className="input" type="number" min="0" step="any" value={tolerance}
                    title="刚体容差（mm，须大于 0）"
                    onChange={(event) => onTolerance(event.target.value)} /></label>
        </div>
        {(!jitterValid || !toleranceValid) && <p className="callout" role="alert">请输入有限数值：抖动修正不小于 0，刚体容差大于 0。</p>}
    </div>;
}
