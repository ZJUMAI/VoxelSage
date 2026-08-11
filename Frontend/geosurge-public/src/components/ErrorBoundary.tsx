import React from 'react';
import { AlertTriangle, RefreshCw, Activity } from 'lucide-react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[ErrorBoundary] 捕获渲染异常:', error);
    console.error('[ErrorBoundary] 组件栈:', errorInfo.componentStack);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  handleFullReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      const errMsg = this.state.error?.message || '未知错误';
      const errStack = this.state.error?.stack || '';

      return (
        <div className="h-screen w-screen flex items-center justify-center bg-slate-950 text-slate-100 font-sans p-6">
          <div className="max-w-lg w-full bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden">
            {/* Header */}
            <div className="flex items-center gap-3 px-6 py-4 bg-slate-850 border-b border-slate-800">
              <div className="w-10 h-10 rounded-xl bg-red-500/20 border border-red-500/30 flex items-center justify-center">
                <AlertTriangle size={20} className="text-red-400" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-slate-100 font-mono uppercase tracking-wide">
                  页面渲染异常
                </h2>
                <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                  GeoSurge 遇到了一个意外错误
                </p>
              </div>
            </div>

            {/* Body */}
            <div className="px-6 py-5 space-y-4">
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-xs space-y-2">
                <div className="flex items-center gap-2 text-red-400 font-semibold">
                  <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                  <span>错误信息</span>
                </div>
                <p className="text-slate-300 text-[11px] leading-relaxed break-words">
                  {errMsg}
                </p>
                {errStack && (
                  <details className="mt-2">
                    <summary className="text-[9px] text-slate-500 cursor-pointer hover:text-slate-300 select-none">
                      查看详细堆栈
                    </summary>
                    <pre className="mt-1 text-[8px] text-slate-500 leading-relaxed max-h-32 overflow-y-auto scrollbar-thin whitespace-pre-wrap">
                      {errStack}
                    </pre>
                  </details>
                )}
              </div>

              <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-3.5 text-xs text-amber-300/90 leading-relaxed">
                💡 这可能是由于后端返回了异常数据或网络连接波动导致的。
                点击「重试」可尝试恢复界面；如果问题持续，请检查后端服务是否正常运行。
              </div>
            </div>

            {/* Actions */}
            <div className="px-6 py-4 bg-slate-850 border-t border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2 text-[9px] text-slate-500 font-mono">
                <Activity size={10} className="animate-pulse" />
                <span>系统状态: 已崩溃 (渲染异常)</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={this.handleRetry}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-bold transition-all shadow-lg shadow-blue-600/20 flex items-center gap-1.5"
                >
                  <RefreshCw size={12} />
                  <span>重试</span>
                </button>
                <button
                  onClick={this.handleFullReload}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition-all border border-slate-700 flex items-center gap-1.5"
                >
                  刷新页面
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
