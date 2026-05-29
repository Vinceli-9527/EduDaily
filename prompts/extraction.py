"""Extraction prompt templates for DeepSeek API — education domain."""

EXTRACTION_SYSTEM = """你是一个专业的中文教育资讯数据抽取专家。你的任务是从非结构化的中文教育文本中抽取出结构化的关键字段。

你必须遵守以下规则：
1. 只抽取文本中明确出现的信息，不要推测或编造数据
2. 如果某个字段在文本中不存在，将其值设为 null
3. 日期字段必须统一为 YYYY-MM-DD 格式
4. 对每个抽取结果提供一个 confidence_score (0.0-1.0)，表示你对该字段抽取的置信度
5. keywords 必须是字符串数组格式
6. 必须返回合法的 JSON，不要包含任何 JSON 之外的文本"""

EXTRACTION_USER_TEMPLATE = """请从以下文本中抽取关键教育资讯信息：

---
{chunk_text}
---

请以 JSON 格式返回抽取结果，字段说明如下：
- policy_name: 政策/法规名称 (string or null)
- policy_level: 政策级别，如 national/province/city/school (string or null)
- education_stage: 学段，如 preschool/primary/junior/senior/higher/vocational (string or null)
- subject_area: 学科，如 math/chinese/english/physics (string or null)
- institution_name: 机构/学校名称 (string or null)
- person_name: 关键人物姓名 (string or null)
- event_date: 事件日期，YYYY-MM-DD格式 (string or null)
- reform_type: 改革类型，如 curriculum/exam/enrollment/teacher (string or null)
- impact_summary: 对教育工作者影响简述，不超过50字 (string or null)
- region: 地区，如北京/上海/广东 (string or null)
- keywords: 关键词列表 (list of strings or null)
- confidence_score: 整体抽取置信度 (number, 0.0-1.0)

抽取示例：
文本："2024年9月，教育部发布《关于深化高中课程改革的指导意见》，要求全国高中从2025年秋季起逐步实施新课程方案。"
输出：{{"policy_name":"关于深化高中课程改革的指导意见","policy_level":"national","education_stage":"senior","subject_area":null,"institution_name":"教育部","person_name":null,"event_date":"2024-09","reform_type":"curriculum","impact_summary":"全国高中将实施新课程方案","region":null,"keywords":["课程改革","高中","新课程方案"],"confidence_score":0.9}}

仅返回 JSON，不要输出其他内容："""


def build_extraction_messages(chunk_text: str) -> list[dict]:
    """Build the message list for an extraction API call."""
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(chunk_text=chunk_text)},
    ]
