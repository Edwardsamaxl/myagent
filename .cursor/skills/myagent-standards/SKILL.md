---
name: myagent-standards
description: >-
  Applies unified writing and engineering standards for this repository: prose
  quality, Chinese (simplified) user-facing text, code citation format, minimal
  diffs, and documentation structure. Use when writing or editing code, docs,
  README, commits, PR descriptions, or when the user asks for consistency across
  documentation and implementation.
---

# myagent 统一规范

在**写代码**与**写文档**（含注释、提交说明、PR 描述）时，同时遵守本节；冲突时以仓库内更具体的约定为准。

## 语言与受众

- 面向用户的说明、文档正文、对话回复：**简体中文**（技术名词可保留英文）。
- 代码标识符、提交信息 subject 可与团队约定一致；若未约定，提交 subject 可用英文 conventional commits。

## 代码

- **范围**：只改任务所需；避免顺手重构、无关文件、扩大范围。
- **风格**：先读周边代码，匹配命名、类型、导入与注释密度；能复用则复用。
- **引用代码**：在对话里展示现有代码时，使用 Cursor 要求的代码引用格式（` ```startLine:endLine:path ` `），引用块单独成行。
- **质量**：优先清晰默认路径；避免仅为“防御”而堆砌 try/except；非必要不删无关注释。

## 文档与说明文

- **结构**：标题层级清晰；复杂内容可用列表或小标题分段。
- **链接**：引用外部网页用完整 `https://` URL；路径给完整字符串。
- **语气**：完整句、技术博客级清晰度；少堆砌加粗与反引号装饰。
- **范围**：用户未要求时，不主动新增大段独立 markdown 文档文件；若任务明确要求文档再写。

## 一致性自检（短清单）

写完后快速核对：

- [ ] 代码改动是否紧贴需求、无无关 diff？
- [ ] 用户可见文字是否为简体中文（若适用）？
- [ ] 文档与代码中的术语是否一致（同一概念同一叫法）？
- [ ] 是否误改了用户未点名的文件？

## 延伸阅读

更细的条款与示例见 [reference.md](reference.md)。
