from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage

class TranslationAgent:
    """高质量翻译Agent"""
    
    SUPPORTED_LANGUAGES = [
        "中文", "英文", "日文", "韩文", 
        "法文", "德文", "西班牙文", "葡萄牙文"
    ]
    
    def __init__(self, llm):
        self.llm = llm
    
    def translate(
        self,
        content: str,
        source_lang: str,
        target_lang: str,
        tone: str = "neutral"
    ) -> str:
        """高质量翻译"""
        prompt = f"""请将以下内容从{source_lang}翻译为{target_lang}：

原文：
{content}

翻译要求：
1. 保持原文含义准确
2. 符合{target_lang}的表达习惯
3. 语气风格：{tone}
4. 专业术语准确翻译
5. 保持格式和结构不变

直接输出翻译结果，不要解释。"""
        
        response = self.llm.invoke([
            SystemMessage(content="你是一个专业的翻译专家。"),
            HumanMessage(content=prompt)
        ])
        return response.content
    
    def translate_with_context(
        self,
        content: str,
        source_lang: str,
        target_lang: str,
        context: str,
        domain: str = "general"
    ) -> str:
        """带领域上下文的高质量翻译"""
        domain_guide = self._get_domain_guide(domain)
        
        prompt = f"""作为{domain}领域的专业翻译，请将以下内容翻译：

源语言：{source_lang}
目标语言：{target_lang}
领域：{domain}

上下文信息：
{context}

领域翻译指南：
{domain_guide}

原文：
{content}

请输出专业、准确、地道的翻译结果。"""
        
        response = self.llm.invoke([
            SystemMessage(content="你是一个专业的领域翻译专家。"),
            HumanMessage(content=prompt)
        ])
        return response.content
    
    def _get_domain_guide(self, domain: str) -> str:
        """获取领域翻译指南"""
        guides = {
            "tech": "- 技术术语使用标准翻译\n- 保持技术准确性\n- 可适当保留英文缩写",
            "marketing": "- 营销文案本地化\n- 符合当地文化习惯\n- 保持营销感染力",
            "legal": "- 法律术语精确\n- 结构严谨\n- 避免歧义表达",
            "medical": "- 医学术语规范\n- 准确性优先\n- 注明注意事项"
        }
        return guides.get(domain, "一般翻译标准")