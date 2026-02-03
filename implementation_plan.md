# 毕业设计实施方案：基于 LLM 的 BAS 系统

基于文档《LLM 辅助 BAS 攻击载荷研究》的指导思想，我们将本项目从单纯的前端展示扩展为前后端分离的完整 BAS（Breach and Attack Simulation）系统。

## 1. 系统架构设计

遵循文档推荐的 **"规划-生成-验证-修正"** 闭环架构：

### A. 前端 (Frontend) - `Running`
- **技术栈**: React + Vite + TailwindCSS (现有)
- **功能**:
  - **任务编排界面**: 可视化展示渗透测试任务树 (PTT)。
  - **实时监控**: 展示攻击过程、Payload 生成流、WAF 响应状态。
  - **报告生成**: 渲染最终的测试报告。

### B. 后端 (Backend) - `To Be Built`
- **技术栈**: Python (FastAPI)
  - Python 是安全领域（如 Pwntools）和 AI 领域（PyTorch/LangChain）的标准语言，最适合实现文档中的复杂逻辑。
- **核心模块**:
  1.  **Planner (规划层)**: 
      - 实现 `PentestTaskTree` 类，管理测试状态上下文。
      - 负责分解攻击目标（例如：URL -> 端口扫描 -> 目录爆破 -> SQL注入测试）。
  2.  **Generator (生成层)**:
      - **LLM Client**: 集成 OpenAI 接口（如 user 之前配置的 hiapi.online）。
      - **Mutation Engine**: 实现基于变异的 Payload 增强算法 (参考 WAF-A-MoLE)。
      - **RAG Module**: (可选) 挂载向量数据库存储历史 Payload。
  3.  **Verifier (验证层)**:
      - **Request Engine**: 发送 HTTP 请求并捕获响应。
      - **Response Analyzer**: 分析 WAF 拦截代码 (403/WAF指纹) 和 漏洞触发特征。
      - **Headless Browser**: (进阶) 集成 Playwright 进行 XSS 动态验证。
  4.  **Feedback Loop (反馈层)**:
      - 将验证结果结构化（拦截/绕过/成功），反馈给 Generator 进行下一轮 Prompt 优化。

## 2. 目录结构规划

我们将重构项目目录如下：

```
WAF-LLM-Adversarial-Workbench/
├── README.md               # 项目说明
├── implementation_plan.md  # 本实施方案
├── frontend/               # (原根目录的前端文件移动至此)
│   ├── src/
│   ├── vite.config.ts
│   └── ...
├── backend/                # (新建 Python 后端)
│   ├── app/
│   │   ├── main.py         # API 入口
│   │   ├── core/           # 核心逻辑
│   │   │   ├── planner.py  # 规划器
│   │   │   ├── generator.py# 攻击载荷生成
│   │   │   └── verifier.py # 智能验证
│   │   └── api/            # 路由定义
│   ├── requirements.txt
│   └── .env
└── docker/                 # (可选) 靶场和部署配置
```

## 3. 第一阶段实施步骤 (MVP)

1.  **环境重构**: 将现有前端代码移动到 `frontend/` 目录，保持根目录整洁。
2.  **后端搭建**: 初始化 FastAPI 项目，配置 LLM 连接。
3.  **核心原型**:
    - 实现一个简单的 `Generator`，能调用 LLM 生成 SQL 注入 Payload。
    - 实现一个简单的 `Verifier`，能对目标 URL 发送请求并判断是否被 WAF 拦截。
4.  **前后端联调**: 前端通过 API 触发扫描任务，实时显示日志。

## 4. 关键技术点实现 (对应文档)

- **PTT (任务树)**: 在后端维护一个内存中的 `TaskGraph` 对象。
- **AutoPentester 策略**: 在 `planner.py` 中编写 Prompt 模板，让 LLM 充当 "策略分析器"。
- **智能验证**: 在 `verifier.py` 中引入简单的响应特征匹配（后续升级为 Headless Browser）。
