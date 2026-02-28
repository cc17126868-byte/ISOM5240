pip install -r requirements.txt
from transformers import pipeline
import streamlit as st
import pandas as pd

# 使用Streamlit的缓存机制加载模型（避免重复加载）
@st.cache_resource
def load_model(model_name):
    """加载并缓存情感分析模型"""
    try:
        # return_all_scores=True 返回所有标签的置信度
        return pipeline("sentiment-analysis", 
                       model=model_name, 
                       return_all_scores=True)
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

def analyze_sentiment(text, classifier):
    """分析文本情感"""
    if not text.strip():
        return None
    try:
        result = classifier(text)
        return result[0]  # 返回分析结果
    except Exception as e:
        st.error(f"分析出错: {str(e)}")
        return None

def get_sentiment_emoji(label):
    """情感标签转Emoji"""
    emoji_map = {
        'POSITIVE': '😊',
        'NEGATIVE': '😞',
        'NEUTRAL': '😐',
    }
    return emoji_map.get(label.upper(), '🤔')

def display_results(results, text_input=None):
    """展示分析结果"""
    if not results:
        return
    
    st.subheader("📊 分析结果")
    
    # 找出置信度最高的情感
    best_result = max(results, key=lambda x: x['score'])
    
    # 使用两列布局展示主要指标
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "预测情感",
            f"{get_sentiment_emoji(best_result['label'])} {best_result['label']}",
            f"{best_result['score']:.2%}"
        )
    with col2:
        st.metric("置信度", f"{best_result['score']:.2%}")
    
    # 展示详细评分表格
    st.subheader("📋 详细评分")
    df = pd.DataFrame(results)
    df['score'] = df['score'].apply(lambda x: f"{x:.2%}")
    df['emoji'] = df['label'].apply(get_sentiment_emoji)
    df = df[['emoji', 'label', 'score']]
    df.columns = ['', '情感', '置信度']
    st.dataframe(df, use_container_width=True, hide_index=True)
