# MEMORY

## 用户偏好
- 用户信任我的判断，可在说明方案后直接执行
- 用户配置了丰富的 agent 库在 `C:/Users/Ed/.claude/agents/`，工作时按需加载

## 可用 Agent 索引

工作时遇到以下类型任务，读取对应 agent 文件并以其角色执行：

| 任务类型 | Agent 文件 | 用途 |
|---------|-----------|------|
| 前端开发/React/PWA | `engineering-frontend-developer.md` | UI框架、性能、可访问性 |
| AI/ML/RAG/LLM | `engineering-ai-engineer.md` | 模型、向量检索、ML系统 |
| 代码审查 | `engineering-code-reviewer.md` | 安全、正确性、可维护性 |
| 系统架构/DDD | `engineering-software-architect.md` | 架构决策、ADRs、边界设计 |
| 后端/数据库/API | `engineering-backend-architect.md` | 可扩展性、安全、API设计 |
| 高级全栈开发 | `engineering-senior-developer.md` | Premium实现、Laravel/Three.js |
| DevOps/SRE | `engineering-sre.md` | 监控、部署、可靠性 |
| 产品管理 | `product-manager.md` | 需求分析、优先级、PRD |
| 安全工程 | `engineering-security-engineer.md` | 渗透测试、威胁建模、安全审计 |
| 代码生成/实现 | `engineering-rapid-prototyper.md` | 快速原型、MVP |

Agent 文件路径：`C:/Users/Ed/.claude/agents/<name>.md`

**加载方式**：遇到相关任务时，用 `Read` 加载对应 agent 文件，以其角色和思维方式工作。

## 重要决策

## 长期目标
