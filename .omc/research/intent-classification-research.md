# Intent Classification 最佳实践调研报告

## 概述

本报告调研了意图分类（Intent Classification）领域的最新研究进展和开源项目，重点关注 Few-shot/Few-shot Learning、Zero-shot Classification、Hierarchical Intent Classification 以及 LLM-based Intent Classification 方向。

---

## 一、核心方法论总结

### 1. AutoIntent (EMNLP 2025) — AutoML for Intent Classification

**来源**: [deeppavlov/autointent](https://github.com/deeppavlov/autointent)

**核心思路**:
- 提供一个 AutoML 框架，自动搜索最优的意图分类 pipeline 配置
- 支持 sklearn-like 的 `fit-predict` 接口，降低使用门槛
- 内置多种分类器、embedder、verbalizer 的组合

**优缺点**:
- **优点**: 开箱即用，自动调优，适合没有深度学习经验的团队
- **缺点**: 自动化搜索带来额外计算开销，框架灵活性受限

**适用场景**: 快速原型验证、多分类场景、离线 AutoML 搜索

**可参考的设计模式**:
- Pipeline preset 机制（`Pipeline.from_preset("classic-light")`）
- 模块化分类器注册表

---

### 2. FCSLM: Collaborating Small and Large Models (EMNLP Findings 2025)

**来源**: [EMNLP 2025 Findings](https://aclanthology.org/2025.findings-emnlp.749.pdf)

**核心思路**:
- **小模型**（PLM-based）负责快速分类
- **大模型**（LLM）负责数据增强 + Out-of-Scope 检测
- LLM 通过 paraphrase 生成更多训练数据，同时利用对比学习减少标签干扰

**优缺点**:
- **优点**: 兼顾推理速度和精度；OOS 检测能力强
- **缺点**: 需要同时部署小模型和大模型，架构复杂

**适用场景**: 少样本（Few-shot）场景、同时需要分类和 OOS 检测的生产系统

**可参考的设计模式**:
- LLM + 小模型协同架构
- 基于对比学习的标签空间缩减策略
- In-context example retrieval + dynamic label refinement

---

### 3. REIC: RAG-Enhanced Intent Classification (EMNLP Industry 2025)

**来源**: [EMNLP 2025 Industry Track](https://aclanthology.org/2025.emnlp-industry.74/)

**核心思路**:
- 将 RAG（Retrieval-Augmented Generation）引入意图分类
- 通过检索相关示例增强分类器的上下文理解
- 动态融合检索结果 + 微调 LLM，提升大规模多意图分类效果

**优缺点**:
- **优点**: 在大规模客服场景下显著优于纯微调或纯 Prompt 方法；无需频繁重训练
- **缺点**: 依赖检索系统质量；增加系统复杂度

**适用场景**: 大规模客服意图分类、多垂直领域（multi-vertical）场景

**可参考的设计模式**:
- RAG + LLM 分类的混合架构
- Intent Probability 计算的置信度机制
- 动态示例检索增强分类

---

### 4. TEXTOIR: Open Intent Recognition Toolkit (ACL 2021)

**来源**: [thuiar/TEXTOIR](https://github.com/thuiar/TEXTOIR) (243 stars)

**核心思路**:
- 开源工具包，集成 Open Intent Detection 和 Open Intent Discovery 两大任务
- 支持多种 SOTA 算法：ADB (AAAI 2021)、DA-ADB (IEEE/ACM TASLP 2023)、USNID (IEEE TKDE 2023)
- 提供统一的数据接口和可视化平台

**优缺点**:
- **优点**: 算法覆盖全面，适合研究；提供标准化 Benchmark
- **缺点**: 偏向研究而非生产；部分代码年久失修

**适用场景**: 学术研究、Open-set Intent Detection、新意图发现

**可参考的设计模式**:
- 统一的数据接口设计（`Dataset` 抽象）
- 算法模块化 + 可视化平台
- 决策边界自适应调整（ADB）

---

### 5. Joint Intent + Slot Filling (BERT-based)

**来源**: [Linear95/bert-intent-slot-detector](https://github.com/Linear95/bert-intent-slot-detector) (239 stars)

**核心思路**:
- 利用 BERT 的 `[CLS]` token 预测 intent
- 利用序列标注（token-level hidden states）预测 slot values
- 联合训练，共享 BERT encoder

**优缺点**:
- **优点**: 训练和推理速度快，适合端侧部署；联合建模提升效果
- **缺点**: 层级标签处理能力有限；不适合超大规模意图集

**适用场景**: 资源受限的端侧部署、Intent + Slot 联合任务

**可参考的设计模式**:
- `[CLS]` 作为句子级表示用于分类
- 多任务学习（Intent Classification + Slot Filling）
- Joint 模型设计

---

## 二、Hierarchical Intent Classification 方法

### HYDRA (EMNLP 2025)

**来源**: [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.472.pdf)

**核心思路**:
- 多头分类器架构，每个层级一个 head，共享 BERT encoder
- 三种配置：Local Heads Only / Local + Global Head / Local + Nested Head
- 将层级文本分类建模为多任务学习问题

**关键设计**:
```
Local Heads Only: 每个层级独立的 MLP 分类器
Local + Global Head: 添加全局分类头对齐层级间表示
Local + Nested Head: 全局头以局部头的输出为输入
```

**适用场景**: 层级结构明显的意图分类（如：订单 > 退款 > 物流问题）

---

### PAAM-HiA-T5 / HiA-T5

**来源**: [COLING 2022](https://aclanthology.org/2022.coling-1.95.pdf)

**核心思路**:
- 基于 T5 的层级感知模型
- PAAM（Pattern-aware Adaptive Masking）模块捕捉层级依赖
- 自上而下逐层预测，结合父标签信息辅助子标签分类

**可参考的设计模式**:
- 层级感知的 label embedding 初始化
- 父子标签依赖建模
- 自上而下贪婪预测 + 层级损失加权

---

## 三、Zero-shot / Few-shot 方法

### ServiceNow ACL Industry 2023 研究

**来源**: [ACL Anthology 2023](https://aclanthology.org/2023.acl-industry.71/)

**对比的四种方法**:
1. **Domain Adaptation**: 在相关领域数据上预训练
2. **Data Augmentation**: paraphrasing、back-translation
3. **Zero-shot LLM Prompting**: 使用 intent descriptions 作为 prompt
4. **Parameter-Efficient Fine-tuning (T-few on Flan-T5)**: 结果最优

**关键结论**:
- T-few (LoRA/adapter-based) 在 1-shot 时就已超越 zero-shot GPT-3
- Zero-shot prompting with intent descriptions 非常具有竞争力
- **推荐策略**: 优先使用 instruction-finetuned 小模型（如 Flan-T5-XL）配合 LoRA

---

### Dynamic Label Refinement (ACL Short 2025)

**来源**: [ACL 2025 Short](https://aclanthology.org/2025.acl-short.3/)

**核心思路**:
- 使用 LLM 动态细化意图标签（基于语义理解）
- 从训练集中为测试输入检索相关示例
- 通过 in-context learning 解决相似意图之间的语义重叠问题

---

## 四、生产系统设计建议

### 架构分层

```
┌─────────────────────────────────────────────┐
│            Intent Router (LLM)               │  ← Zero-shot / Few-shot 兜底
├─────────────────────────────────────────────┤
│         Primary Classifier (Fine-tuned PLM)  │  ← BERT/Sentence-BERT
├─────────────────────────────────────────────┤
│           RAG Enhancement Layer              │  ← 示例检索 + 上下文增强
├─────────────────────────────────────────────┤
│           OOS Detector                       │  ← 置信度阈值 / 专门 OOS 模型
└─────────────────────────────────────────────┘
```

### 阈值设定策略

- **置信度阈值**: 推荐 0.7-0.85 范围，低于阈值则触发 OOS 或 LLM 路由
- **Temperature 设定**: 推理时推荐 temperature=0.1（确定性输出）
- **Top-K 过滤**: 只考虑 top-k 候选意图，减少误分类

### 特征选择

- **推荐**: Sentence-BERT (all-mpnet-base-v2) 作为 embedding 模型
- **备选**: BGE-large-en-v1.5（中文支持更好）
- **端侧**: DistilBERT + 量化（int8）

### 多意图检测

- 使用多标签分类（sigmoid 而非 softmax）
- 设定意图数量上限（如最多 3 个并发意图）
- 结合层级结构：先预测高层意图，再预测低层子意图

---

## 五、可落地设计建议

1. **小样本场景**: 优先使用 FCSLM 思路，小模型 + LLM 协同，1-3 shots 即可取得良好效果

2. **大规模多意图**: 采用 REIC 架构，RAG 增强 + 微调 LLM，避免频繁重训练

3. **层级意图结构**: 使用 HYDRA 的多 head 架构，或 HiA-T5 的自上而下预测

4. **快速原型**: 使用 AutoIntent 的 AutoML 框架快速验证baseline

5. **生产部署**: 推荐 BERT-base + Joint Intent/Slot 联合模型，配合 OOS 检测层

---

## 六、关键参考资源

| 资源 | 类型 | 地址 |
|------|------|------|
| AutoIntent | GitHub | https://github.com/deeppavlov/autointent |
| TEXTOIR | GitHub | https://github.com/thuiar/TEXTOIR |
| bert-intent-slot-detector | GitHub | https://github.com/Linear95/bert-intent-slot-detector |
| open-intent-classifier | GitHub | https://github.com/SerjSmor/open-intent-classifier |
| FCSLM | Paper | https://aclanthology.org/2025.findings-emnlp.749/ |
| REIC | Paper | https://aclanthology.org/2025.emnlp-industry.74/ |
| HYDRA | Paper | https://aclanthology.org/2025.emnlp-main.472.pdf |
| Zero/Few-shot ACL 2023 | Paper | https://aclanthology.org/2023.acl-industry.71/ |
| Dynamic Label Refinement | Paper | https://aclanthology.org/2025.acl-short.3/ |
