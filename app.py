import streamlit as st
import pickle
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import os


# ---------------------------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# CUSTOM CSS FOR ATTRACTIVE UI
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 20px 0;
    }
    
    .sub-header {
        text-align: center;
        color: #ffffff;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Text area - FIXED for better visibility */
    textarea {
        border-radius: 15px !important;
        border: 3px solid rgba(255, 255, 255, 0.5) !important;
        background: rgba(255, 255, 255, 0.98) !important;
        font-size: 1.05rem !important;
        padding: 20px !important;
        color: #1a1a1a !important;
        font-family: 'Georgia', serif !important;
        line-height: 1.6 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
    }
    
    textarea::placeholder {
        color: #666666 !important;
        opacity: 0.7 !important;
    }
    
    textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.5) !important;
        background: rgba(255, 255, 255, 1) !important;
    }
    
    /* Label for text area */
    .stTextArea label {
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
    }
    
    /* Result boxes */
    .result-box-real {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(81, 207, 102, 0.4);
        margin: 20px 0;
        animation: slideIn 0.5s ease-out;
    }
    
    .result-box-fake {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(255, 107, 107, 0.4);
        margin: 20px 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .result-subtitle {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-bottom: 20px;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 1.8rem;
        font-weight: bold;
        backdrop-filter: blur(10px);
        margin-top: 10px;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.15);
        border-left: 5px solid #ffffff;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }
    
    .debug-box {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 20px;
        color: #ffffff;
        font-family: 'Courier New', monospace;
        margin: 20px 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 50px;
        padding: 30px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #51cf66, #37b24d);
        border-radius: 10px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: white !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered News Authenticity Analyzer</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        with open("mlp.pkl", "rb") as f:
            mlp = pickle.load(f)
        w2v = Word2Vec.load("word2vec.model")
        return mlp, w2v
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None, None

mlp, w2v = load_models()

if mlp is None or w2v is None:
    st.error("⚠️ Models could not be loaded. Please check the model files.")
    st.stop()


# ---------------------------------------------------------------------------
# VECTOR FUNCTION
# ---------------------------------------------------------------------------
def get_avg_word2vec(text, model):
    tokens = simple_preprocess(text)
    word_vecs = [model.wv[w] for w in tokens if w in model.wv.index_to_key]
    
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size), 0, 0
    
    return np.mean(word_vecs, axis=0), len(tokens), len(word_vecs)


# ---------------------------------------------------------------------------
# UI INPUT
# ---------------------------------------------------------------------------
st.markdown("")
news_text = st.text_area(
    "📝 Paste Your News Article Here",
    height=350,
    placeholder="Copy and paste the complete news article text here for analysis. Include the full content for best results...",
)

st.markdown("")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    analyze = st.button("🔍 Analyze Article")

with col2:
    clear = st.button("🗑️ Clear Text")

if clear:
    st.rerun()


# ---------------------------------------------------------------------------
# ANALYSIS BLOCK
# ---------------------------------------------------------------------------
if analyze:
    if not news_text.strip():
        st.markdown("""
            <div class="info-box">
                <strong>⚠️ No Input Detected</strong><br>
                Please paste an article in the text area above to begin analysis.
            </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("🔄 Analyzing article... Please wait..."):
            
            # Get vector
            vector, total_tokens, matched_tokens = get_avg_word2vec(news_text, w2v)
            vector = vector.reshape(1, -1)
            
            # Get probabilities
            proba = mlp.predict_proba(vector)[0]
            fake_prob = proba[0]
            real_prob = proba[1]
            
            # Prediction
            prediction = 1 if real_prob >= 0.50 else 0
            
            # Word count
            word_count = len(news_text.split())
            char_count = len(news_text)
            
        st.markdown("---")
        
        # Results display
        if prediction == 1:
            st.markdown(f"""
                <div class="result-box-real">
                    <div class="result-title">✅ REAL NEWS DETECTED</div>
                    <div class="result-subtitle">This article appears to be legitimate and trustworthy</div>
                    <div class="confidence-badge">{real_prob*100:.1f}% Confidence</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Real Probability", f"{real_prob*100:.1f}%", delta=f"+{(real_prob-fake_prob)*100:.1f}%")
            with col2:
                st.metric("📝 Words Analyzed", word_count)
            with col3:
                st.metric("🎯 Match Rate", f"{matched_tokens/max(total_tokens,1)*100:.0f}%")
            
        else:
            st.markdown(f"""
                <div class="result-box-fake">
                    <div class="result-title">⚠️ FAKE NEWS DETECTED</div>
                    <div class="result-subtitle">This content shows patterns of misinformation</div>
                    <div class="confidence-badge">{fake_prob*100:.1f}% Confidence</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Fake Probability", f"{fake_prob*100:.1f}%", delta=f"+{(fake_prob-real_prob)*100:.1f}%")
            with col2:
                st.metric("📝 Words Analyzed", word_count)
            with col3:
                st.metric("🎯 Match Rate", f"{matched_tokens/max(total_tokens,1)*100:.0f}%")
        
        # Probability breakdown
        st.markdown("### 📈 Detailed Probability Breakdown")
        st.progress(float(real_prob), text=f"🟢 Real News: {real_prob*100:.2f}%")
        st.progress(float(fake_prob), text=f"🔴 Fake News: {fake_prob*100:.2f}%")
        
        # Analysis stats
        st.markdown(f"""
            <div class="info-box">
                <strong>📊 Analysis Statistics</strong><br><br>
                <strong>Total Words:</strong> {word_count} | <strong>Characters:</strong> {char_count}<br>
                <strong>Tokens Processed:</strong> {total_tokens} | <strong>Vocabulary Matches:</strong> {matched_tokens}
            </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
    <div class="footer">
        <p style='font-size: 1.1rem; margin-bottom: 10px;'>⚠️ <strong>Disclaimer</strong></p>
        <p>This tool is for educational and research purposes only. Always verify news through multiple trusted sources.</p>
        <p style='margin-top: 20px; opacity: 0.7;'>Powered by Machine Learning | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)