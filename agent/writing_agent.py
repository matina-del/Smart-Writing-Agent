from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Optional
import json

class WritingAgent:
    """智能写作Agent"""
    
    def __init__(self, llm, templates: dict):
        self.llm = llm
        self.templates = templates
        
    def generate(
        self,
        topic: str,
        style: str = "professional",
        length: str = "medium",
        audience: str = "general",
        **kwargs
    ) -> Dict[str, any]:
        """多阶段内容生成"""
        # 阶段1：大纲生成
        outline = self._generate_outline(topic, length, **kwargs)
        # 阶段2：分段生成
        sections = self._generate_sections(outline, style, audience)
        # 阶段3：整体优化
        final_content = self._optimize_content(sections, style)
        # 阶段4：质量评估
        quality_score = self._evaluate_quality(final_content)
        
        return {
            "outline": outline,
            "content": final_content,
            "quality_score": quality_score,
            "word_count": len(final_content)
        }
    
    def _generate_outline(self, topic: str, length: str, **kwargs) -> Dict:
        """生成文章大纲"""
        length_map = {
            "short": {"sections": 3, "paragraphs": 2},
            "medium": {"sections": 4, "paragraphs": 3},
            "long": {"sections": 6, "paragraphs": 4}
        }
        config = length_map.get(length, length_map["medium"])
        
        prompt = f"""为以下主题生成详细文章大纲：

主题：{topic}

要求：
- 共{config['sections']}个主要章节
- 每个章节{config['paragraphs']}-{config['paragraphs']+1}个段落
- 包含开头的引入和结尾的总结
- 逻辑清晰，层次分明

请以JSON格式输出：
{{
  "title": "文章标题",
  "sections": [
    {{
      "heading": "章节标题",
      "key_points": ["要点1", "要点2"],
      "target_words": 300
    }}
  ],
  "conclusion": "总结要点"
}}"""
        response = self.llm.invoke([
            SystemMessage(content="你是一个专业的文章结构设计师。"),
            HumanMessage(content=prompt)
        ])
        try:
            return json.loads(response.content)
        except:
            return {"sections": []}
    
    def _generate_sections(self, outline: Dict, style: str, audience: str) -> List[str]:
        """根据大纲生成各章节内容"""
        style_guide = self._get_style_guide(style)
        sections = []
        
        for idx, section in enumerate(outline.get("sections", [])):
            section_prompt = f"""作为专业的{style}风格写作者，请撰写以下章节：

章节标题：{section.get('heading', '')}
章节要点：{chr(10).join(section.get('key_points', []))}
目标字数：约{section.get('target_words', 300)}字
目标读者：{audience}

风格指南：
{style_guide}

请撰写完整、流畅、吸引人的内容。"""
            response = self.llm.invoke([
                SystemMessage(content="你是一个专业的内容创作助手。"),
                HumanMessage(content=section_prompt)
            ])
            sections.append(response.content)
        return sections
    
    def _optimize_content(self, sections: List[str], style: str) -> str:
        """整体内容优化"""
        prompt = f"""请对以下文章内容进行整体优化：

1. 确保文章风格一致（{style}风格）
2. 优化段落之间的过渡
3. 消除重复内容
4. 提升语言表达
5. 确保逻辑连贯

文章内容：
{chr(10).join(sections)}

请输出优化后的完整文章。"""
        response = self.llm.invoke([
            SystemMessage(content="你是一个专业的文章编辑。"),
            HumanMessage(content=prompt)
        ])
        return response.content
    
    def _evaluate_quality(self, content: str) -> Dict:
        """质量评估"""
        prompt = f"""请评估以下文章的质量：

{content}

请从以下维度评分（1-10分）：
1. 内容完整性
2. 逻辑清晰度
3. 语言表达
4. 创意程度
5. 读者吸引力

请以JSON格式输出：
{{
  "total_score": 8,
  "breakdown": {{"维度": 8}},
  "strengths": ["优点1", "优点2"],
  "improvements": ["可改进点1", "可改进点2"]
}}"""
        response = self.llm.invoke([
            SystemMessage(content="你是一个专业的内容质量评审。"),
            HumanMessage(content=prompt)
        ])
        try:
            return json.loads(response.content)
        except:
            return {}
    
    def _get_style_guide(self, style: str) -> str:
        """获取风格指南"""
        guides = {
            "professional": """- 使用正式、专业的语言\n- 避免口语化表达\n- 强调逻辑和数据支撑\n- 段落结构清晰""",
            "casual": """- 使用轻松、口语化的语言\n- 可以使用网络用语\n- 语气亲切友好\n- 适当使用感叹句""",
            "young": """- 使用年轻化的表达\n- 融入当下流行元素\n- 语言活泼有活力\n- 适合社交媒体风格"""
        }
        return guides.get(style, guides["professional"])