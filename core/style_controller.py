from enum import Enum
from typing import Dict
from langchain_core.messages import HumanMessage

class WritingStyle(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    YOUNG = "young"
    ACADEMIC = "academic"
    MARKETING = "marketing"

class StyleController:
    """风格控制器"""
    
    STYLE_CONFIGS = {
        WritingStyle.PROFESSIONAL: {
            "tone": "formal",
            "sentence_complexity": "high",
            "vocabulary_level": "advanced",
            "emoji_usage": 0.0,
            "paragraph_length": "medium"
        },
        WritingStyle.CASUAL: {
            "tone": "relaxed",
            "sentence_complexity": "medium",
            "vocabulary_level": "intermediate",
            "emoji_usage": 0.3,
            "paragraph_length": "short"
        },
        WritingStyle.YOUNG: {
            "tone": "energetic",
            "sentence_complexity": "low",
            "vocabulary_level": "mixed",
            "emoji_usage": 0.5,
            "paragraph_length": "short"
        },
        WritingStyle.MARKETING: {
            "tone": "persuasive",
            "sentence_complexity": "medium",
            "vocabulary_level": "accessible",
            "emoji_usage": 0.4,
            "paragraph_length": "short"
        }
    }
    
    def __init__(self, llm):
        self.llm = llm
        self.current_style = WritingStyle.PROFESSIONAL
    
    def set_style(self, style: WritingStyle):
        """设置写作风格"""
        self.current_style = style
    
    def transform(self, content: str, target_style: WritingStyle) -> str:
        """将内容转换为目标风格"""
        config = self.STYLE_CONFIGS[target_style]
        
        prompt = f"""请将以下文章转换为{target_style.value}风格：

原文：
{content}

目标风格特征：
- 语气：{config['tone']}
- 句子复杂度：{config['sentence_complexity']}
- 词汇水平：{config['vocabulary_level']}
- Emoji使用比例：{config['emoji_usage']*100}%
- 段落长度：{config['paragraph_length']}

请保持原文的核心内容，仅调整风格表达方式。"""
        
        return self.llm.invoke([HumanMessage(content=prompt)]).content