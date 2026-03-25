from __future__ import annotations

import re

_FIN_HINT = re.compile(
    r"(营业收入|净利润|每股收益|ROE|净资产|现金流|分红|总资产|负债|同比|环比)"
)
_COMPANY_HINT = re.compile(r"(茅台|五粮液|同花顺|公司|股份)")


def should_clarify_for_finance_without_anchor(message: str) -> bool:
    """财务类问题但缺少常见主体/年份锚点时建议澄清（纯结构规则，无置信度阈值）。"""
    if not _FIN_HINT.search(message):
        return False
    has_year = bool(re.search(r"20\d{2}", message))
    has_company = bool(_COMPANY_HINT.search(message))
    if has_year and has_company:
        return False
    return not (has_year or has_company)


def default_clarify_prompt(message: str) -> str:
    if len(message.strip()) < 4:
        return "您的问题较简短，请补充具体想查询的公司、报告年份或指标名称（例如：贵州茅台 2024 年营业收入）。"
    if _FIN_HINT.search(message) and not re.search(r"20\d{2}", message):
        return "请补充要查询的报告年份或期间（例如 2024 年报、2025 年上半年），以便在文档中定位对应数据。"
    return "为更准确回答，请补充公司名称、报告期或具体指标名称。"
