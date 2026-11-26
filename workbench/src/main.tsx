import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './omp.css';
import Processing from './Processing';

// 单屏工作台：顶栏状态条 + 三列（数据浏览器 / 处理队列 / 运行详情）；FBX 查看为覆盖层。
function Workbench() {
    return <>
        <a className="skip-link" href="#main-content"
            onClick={(event) => { event.preventDefault(); document.getElementById('main-content')?.focus(); }}>
            跳至主内容
        </a>
        <Processing />
    </>;
}

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <Workbench />
    </StrictMode>,
);
