import streamlit as st
from sentiment_utils import load_model, analyze_sentiment, display_results

# 页面配置 - 必须在最前面
st.set_page_config(
    page_title="Transformer情感分析",
    page_icon="🤗",
    layout="centered"
)

def main():
    # 标题和说明
    st.title("🎭 Transformer情感分析App")
    st.markdown("""
    使用预训练的Transformer模型分析文本情感。输入任何文字，AI会告诉你它是积极还是消极，
    并给出置信度分数。
    """)
    
    # 侧边栏：模型选择
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 可供选择的模型列表
        model_options = {
            "快速分析 (DistilBERT)": "distilbert-base-uncased-finetuned-sst-2-english",
            "高精度 (RoBERTa)": "siebert/sentiment-roberta-large-english",
            "多语言支持": "nlptown/bert-base-multilingual-uncased-sentiment",
            "社交媒体文本": "cardiffnlp/twitter-roberta-base-sentiment-latest"
        }
        
        selected_model = st.selectbox(
            "选择模型",
            options=list(model_options.keys()),
            index=1  # 默认选RoBERTa
        )
        
        model_name = model_options[selected_model]
        
        st.markdown("---")
        st.markdown("### ℹ️ 关于模型")
        if selected_model == "快速分析 (DistilBERT)":
            st.info("轻量级模型，速度快，适合实时分析")
        elif selected_model == "高精度 (RoBERTa)":
            st.info("大型模型，精度高，适合重要分析任务")
        elif selected_model == "多语言支持":
            st.info("支持100+种语言，评分1-5星")
        else:
            st.info("专门针对Twitter等社交媒体优化")
    
    # 加载模型（带进度提示）
    with st.spinner(f"正在加载模型: {selected_model}... 首次加载可能需要一分钟"):
        classifier = load_model(model_name)
    
    if classifier is None:
        st.error("模型加载失败，请刷新页面重试")
        return
    
    # 成功提示
    st.success("✅ 模型加载完成！")
    
    # 文本输入区
    text_input = st.text_area(
        label="输入要分析的文本：",
        placeholder="例如：I love this product! It works perfectly.",
        height=150,
        max_chars=1000
    )
    
    # 分析按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 分析情感", 
            type="primary", 
            use_container_width=True
        )
    
    if analyze_button:
        if text_input.strip():
            with st.spinner("AI正在思考中..."):
                results = analyze_sentiment(text_input, classifier)
            
            if results:
                display_results(results, text_input)
                
                # 添加一些趣味性
                st.balloons()
        else:
            st.warning("⚠️ 请输入一些文本")
    
    # 添加示例
    with st.expander("📝 试试这些示例"):
        examples = [
            "I absolutely love this new smartphone! The camera is amazing.",
            "This is the worst customer service I've ever experienced.",
            "The movie was okay, not great but not terrible either.",
            "Thank you for your help! You saved my day."
        ]
        for i, example in enumerate(examples):
            if st.button(f"示例 {i+1}", key=f"example_{i}"):
                # 这里可以用session_state实现自动填充
                st.session_state["example_text"] = example

if __name__ == "__main__":
    main()
