from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ============================================================
# 顶层意图（粗粒度，决定 Agent 路由）
# ============================================================
class IntentTier(str, Enum):
    """粗粒度意图分类，决定 Agent 路由（REIC/HYDRA 启发）。"""

    TOOL_ONLY = "tool_only"       # 纯工具调用
    KNOWLEDGE = "knowledge"        # 知识问答
    MIXED = "mixed"                # 工具+知识混合
    CHITCHAT = "chitchat"          # 闲聊/寒暄
    AMBIGUOUS = "ambiguous"        # 意图模糊，需澄清
    OOS = "oos"                    # Out-of-Scope，超出系统能力


# ============================================================
# 细粒度子意图（决定具体执行路径）
# ============================================================
class SubIntent(str, Enum):
    """细粒度子意图，嵌入 IntentTier 内部，驱动具体执行逻辑。"""

    # TOOL_ONLY 子类
    TIME_QUERY = "time_query"
    CALCULATION = "calculation"
    FILE_READ = "file_read"
    MEMORY_OP = "memory_op"
    SKILL_INVOKE = "skill_invoke"
    SINGLE_TOOL_CALL = "single_tool_call"
    MULTI_STEP_TOOL = "multi_step_tool"

    # KNOWLEDGE 子类
    FINANCIAL_QUERY = "financial_query"
    COMPANY_QUERY = "company_query"
    MARKET_DATA = "market_data"
    GENERAL_FACT = "general_fact"

    # MIXED 子类
    DATA_THEN_ANALYZE = "data_then_analyze"
    REPORT_WITH_CALC = "report_with_calc"

    # CHITCHAT 子类
    GREETING = "greeting"
    SELF_INTRO = "self_intro"
    CASUAL = "casual"

    # OOS 子类
    OFF_TOPIC = "off_topic"
    SENSITIVE = "sensitive"
    UNSAFE = "unsafe"


# ============================================================
# 意图分类来源（用于调试和追踪）
# ============================================================
class IntentSource(str, Enum):
    """分类策略来源，追踪置信度来源以便优化。"""
    RULE = "rule"                 # 规则匹配
    EMBEDDING = "embedding"       # Embedding 相似度
    LLM = "llm"                  # LLM 判断
    FALLBACK = "fallback"        # 降级默认


# ============================================================
# 分类结果
# ============================================================
@dataclass
class IntentContext:
    """会话上下文，用于上下文感知分类。"""
    session_history: list = field(default_factory=list)
    pending_clarification: bool = False
    last_intent: IntentTier | None = None
    domain_streak: int = 0


@dataclass
class IntentResult:
    """完整意图分类结果。"""
    tier: IntentTier                          # 顶层意图（路由用）
    sub: SubIntent                            # 细粒度子意图（执行用）
    confidence: float                         # 0.0-1.0 置信度
    source: IntentSource                      # 分类来源

    # 指代消解结果（上下文感知）
    resolved_query: str | None = None         # 消解后的查询
    resolved_slots: dict[str, str] = field(default_factory=dict)  # 提取的槽位

    # 澄清/拒答
    clarify_prompt: str | None = None
    is_oos: bool = False

    # 用于兼容旧代码（废弃字段）
    intent: str | None = field(default=None, repr=False)
    normalized_query: str | None = field(default=None, repr=False)
    slots: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # 兼容旧代码：将 tier 作为 intent 字符串暴露
        if self.intent is None:
            object.__setattr__(self, "intent", self.tier.value)
        if self.normalized_query is None:
            object.__setattr__(self, "normalized_query", self.resolved_query)
        if not self.slots:
            object.__setattr__(self, "slots", self.resolved_slots)


# ============================================================
# 向量检索用的意图描述（REIC 启发）
# ============================================================
INTENT_DESCRIPTIONS: dict[SubIntent, str] = {
    # TOOL_ONLY
    SubIntent.TIME_QUERY: "询问当前时间或日期，如'现在几点'、'今天是几号'",
    SubIntent.CALCULATION: "需要进行数学计算或数值运算，如'加总'、'增长率'",
    SubIntent.FILE_READ: "读取文件内容或打开工作区文件",
    SubIntent.MEMORY_OP: "读写长期记忆或保存信息，如'记住'、'查看记忆'",
    SubIntent.SKILL_INVOKE: "调用已保存的技能或工具",
    SubIntent.SINGLE_TOOL_CALL: "单步工具调用请求",
    SubIntent.MULTI_STEP_TOOL: "需要多个工具配合完成的任务",

    # KNOWLEDGE
    SubIntent.FINANCIAL_QUERY: "查询公司财务数据，如营业收入、净利润、每股收益、ROE等",
    SubIntent.COMPANY_QUERY: "查询公司基本信息，如公司介绍、股东、子公司等",
    SubIntent.MARKET_DATA: "查询市场数据、股价、交易量等",
    SubIntent.GENERAL_FACT: "查询通用知识或事实，不涉及具体公司或财务",

    # MIXED
    SubIntent.DATA_THEN_ANALYZE: "先检索数据再进行分析计算",
    SubIntent.REPORT_WITH_CALC: "生成报告并包含数值计算",

    # CHITCHAT
    SubIntent.GREETING: "问候语，如'你好'、'早上好'",
    SubIntent.SELF_INTRO: "询问AI自身身份或能力",
    SubIntent.CASUAL: "随意闲聊，不涉及具体任务",

    # OOS
    SubIntent.OFF_TOPIC: "超出系统能力范围的请求",
    SubIntent.SENSITIVE: "涉及敏感内容的请求",
    SubIntent.UNSAFE: "不安全或有害的请求",
}


# ============================================================
# 旧版 IntentKind 兼容（后续可删除）
# ============================================================
class IntentKind(str, Enum):
    """对话意图（规则分类初版，后续可接模型 JSON）。"""

    KNOWLEDGE_CORPUS = "knowledge_corpus"
    TOOL_ONLY = "tool_only"
    MIXED = "mixed"
    CHITCHAT = "chitchat"
    AMBIGUOUS = "ambiguous"
    UNSAFE_OR_REFUSE = "unsafe_or_refuse"

    @classmethod
    def from_tier(cls, tier: IntentTier) -> IntentKind:
        """从 IntentTier 映射到旧版 IntentKind（兼容用）。"""
        mapping = {
            IntentTier.TOOL_ONLY: cls.TOOL_ONLY,
            IntentTier.KNOWLEDGE: cls.KNOWLEDGE_CORPUS,
            IntentTier.MIXED: cls.MIXED,
            IntentTier.CHITCHAT: cls.CHITCHAT,
            IntentTier.AMBIGUOUS: cls.AMBIGUOUS,
            IntentTier.OOS: cls.UNSAFE_OR_REFUSE,
        }
        return mapping.get(tier, cls.KNOWLEDGE_CORPUS)


def intent_result_to_legacy(result: IntentResult) -> IntentResult:
    """将新版 IntentResult 转换为旧版兼容格式（保留 intent 字段）。"""
    legacy = IntentResult(
        tier=result.tier,
        sub=result.sub,
        confidence=result.confidence,
        source=result.source,
        resolved_query=result.resolved_query,
        resolved_slots=result.resolved_slots,
        clarify_prompt=result.clarify_prompt,
        is_oos=result.is_oos,
    )
    # 旧版 intent 字段映射
    object.__setattr__(legacy, "intent", IntentKind.from_tier(result.tier).value)
    object.__setattr__(legacy, "normalized_query", result.resolved_query)
    object.__setattr__(legacy, "slots", result.resolved_slots)
    return legacy
