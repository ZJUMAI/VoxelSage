# GeoSurge 智能影像辅助诊断平台

基于多模态 3D 医疗影像与 CT 切片分析的「感知-理解」智能临床辅助诊断平台前端。

> 本项目仅用于医学科研与决策辅助，不可直接作为临床治疗决策依据。

## 功能特性

- **多模态影像上传**：支持 NIfTI (.nii.gz) 与 DICOM 影像的上传与本地解析
- **2D 切片查看器**：轴位切片浏览、切片定位、HU 值标注
- **3D 重建渲染**：器官分割结果的三维重建与交互式查看
- **技能化诊断工作流**：通过对话式 Agent 调用影像分析技能（肝脏综合分析、关键切片选取、肿瘤直径测量、肿瘤-血管距离、血管体积等）
- **多轮对话**：WebSocket 实时流式响应，支持中断与继续
- **患者档案管理**：病例创建、编辑、检索与本地持久化

## 技术栈

- React 19 + TypeScript + Vite 6
- Tailwind CSS 4
- lucide-react 图标
- nifti-reader-js 本地 NIfTI 解析

## 快速开始

```bash
npm install
npm run dev
```

前端默认监听 `http://localhost:3000`。

### 后端依赖

前端通过 HTTP + WebSocket 与后端 Agent 服务通信，后端地址集中配置在 `src/api.ts`：

```ts
export const WS_CHAT_URL = 'ws://localhost:8900/ws/frontend/chat';
export const HTTP_BASE_URL = 'http://localhost:8900';
export const SKILLS_BASE_URL = 'http://localhost:8765';
```

部署时请将这些常量改为你的实际后端地址。详细说明见 `.env.example`。

## 构建

```bash
npm run build   # 产物输出到 dist/
```

## 项目结构

```
src/
├── api.ts                      # 后端地址与上传/健康检查
├── App.tsx                     # 主应用（布局、状态、WebSocket 编排）
├── components/
│   ├── ChatSection.tsx         # 对话界面与技能结果导航
│   ├── RendererPanel.tsx       # 右侧渲染面板（2D/3D/技能结果）
│   ├── SkillResultRenderer.tsx # 技能结果渲染分发
│   ├── SliceViewer.tsx         # 2D 切片查看器
│   ├── DeepSeekSidebar.tsx     # 左侧病例档案栏
│   └── ...
└── utils/
    ├── niftiLoader.ts          # NIfTI 本地解析
    ├── portBUrlHelper.ts       # 后端相对路径 → 前端 URL
    └── ...
```

