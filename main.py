import os
from langchain_openai import ChatOpenAI


from agent.writing_agent import WritingAgent
from core.style_controller import StyleController, WritingStyle
from agent.translation_agent import TranslationAgent


os.environ["OPENAI_API_KEY"] = "sk-真实API密钥填在这里"
def main():
    print("🤖 系统启动：正在初始化智能写作引擎...\n")
    
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        base_url='https://api.gptsapi.net/v1' 
    )
    
    
    writer = WritingAgent(llm=llm, templates={})
    stylist = StyleController(llm=llm)
    translator = TranslationAgent(llm=llm)
    
    topic = "新款智能手表测评，颜值高，心率准"
    
    try:
        print("====== 🚀 阶段1：【生成层】工作开始 ======")
        print("正在生成结构化大纲与初稿...")
        
        draft_result = writer.generate(topic=topic, length="short")
        base_draft = draft_result['content']
        print("✅ 初稿生成完毕！\n")
        
        print("====== 🎨 阶段2：【控制层】工作开始 ======")
        print("正在将初稿注入【小红书营销】风格参数...")
        # 调用风格控制器，传入营销枚举值
        marketing_text = stylist.transform(base_draft, WritingStyle.MARKETING)
        print(f"📌 【最终中文爆款】:\n{marketing_text}\n")
        
        print("====== 🌍 阶段3：【多语言模块】工作开始 ======")
        print("正在进行出海本地化，分发日文版本...")
        # 调用翻译 Agent
        jp_text = translator.translate(marketing_text, source_lang="中文", target_lang="日文", tone="young")
        print(f"📌 【日文分发版本】:\n{jp_text}\n")
        
        

    except Exception as e:
        print(f"❌ 运行中断: {e}")

if __name__ == "__main__":
    main()