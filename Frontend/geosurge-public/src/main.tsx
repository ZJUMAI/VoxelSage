import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import {ErrorBoundary} from './components/ErrorBoundary.tsx';
import './index.css';

// ── 全局 Promise 异常兜底 ──
// 拦截浏览器扩展触发的 false-positive "未捕获" 错误（如
// "A listener indicated an asynchronous response..."），
// 仅以 console.debug 输出，不污染开发者控制台。
window.addEventListener('unhandledrejection', (ev: PromiseRejectionEvent) => {
  const msg = ev.reason?.message || String(ev.reason);
  if (
    msg.includes('listener indicated an asynchronous response') ||
    msg.includes('message channel closed')
  ) {
    // 浏览器扩展通信超时 — 非应用错误，静默抑制
    ev.preventDefault();
    console.debug('[扩展通信]', msg);
    return;
  }
  // 其他未捕获 Promise 错误仍然打印警告
  console.warn('[未捕获 Promise]', ev.reason);
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
