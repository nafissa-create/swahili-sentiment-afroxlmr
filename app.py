import streamlit as st
from transformers import pipeline

MODEL_ID = "nafissaoued/swahili-sentiment-afroxlmr"

st.set_page_config(
    page_title="Kiswahili Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
)

# Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background: #f6f8f7; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
.hero { background: linear-gradient(120deg, #102f2c 0%, #1b5146 100%); padding: 2.2rem 2.4rem; border-radius: 24px; color: white; margin-bottom: 1.3rem; }
.hero-kicker { color: #a9ddc6; letter-spacing: .12em; font-weight: 700; font-size: .78rem; text-transform: uppercase; }
.hero h1 { color: white; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.12; margin: .5rem 0; }
.hero p { color: #d4e5df; font-size: 1.05rem; max-width: 700px; }
.panel { background: white; padding: 1.4rem 1.5rem; border: 1px solid #e5ece8; border-radius: 18px; }
.result-card { background: #ffffff; border: 1px solid #e5ece8; border-left: 6px solid #23785d; border-radius: 18px; padding: 1.5rem; }
.result-label { font-family: 'Manrope', sans-serif; font-size: 1.8rem; font-weight: 800; color: #163b34; }
.metric-card { background: white; border: 1px solid #e5ece8; border-radius: 16px; padding: 1rem; }
.small-muted { color: #687b75; font-size: .92rem; }
div.stButton > button[kind="primary"] { background: #1c6852; border: 1px solid #1c6852; border-radius: 12px; font-weight: 700; min-height: 3rem; }
div.stButton > button { border-radius: 12px; }
hr { border-color: #e4ebe7; }
</style>
""", unsafe_allow_html=True)

# Model 
@st.cache_resource(show_spinner="Loading the sentiment model… the first load may take a moment.")
def load_model():
    return pipeline("text-classification", model=MODEL_ID, top_k=None)

EXAMPLES = {
    "Positive": "Huduma ilikuwa nzuri sana, asante!",
    "Negative": "Hii ni mbaya sana, sijafurahi",
    "Neutral": "Mkutano utafanyika kesho saa tatu",
}

LABEL_META = {
    "positive": {"emoji": "😊", "color": "#23785d", "description": "The text is predicted to express a positive sentiment."},
    "negative": {"emoji": "☹️", "color": "#b34b48", "description": "The text is predicted to express a negative sentiment."},
    "neutral": {"emoji": "😐", "color": "#657b83", "description": "The text is predicted to express a neutral sentiment."},
}

# Header
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Kiswahili AI · NLP Project</div>
  <h1>Understand the sentiment<br>behind Kiswahili text.</h1>
  <p>Explore how AI interprets sentiment in Kiswahili comments, messages and tweets.<br>
  Classify text as positive, negative or neutral.</p>
</div>
""", unsafe_allow_html=True)

# Analyzer 
st.markdown("### Sentiment analyzer")
st.markdown("<p class='small-muted'>Enter a Kiswahili sentence or choose an example to get started.</p>", unsafe_allow_html=True)

if "text" not in st.session_state:
    st.session_state.text = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

cols = st.columns(3)
for col, (label, sentence) in zip(cols, EXAMPLES.items()):
    with col:
        if st.button(f"Try {label} example", use_container_width=True, key=f"example_{label}"):
            st.session_state.text = sentence
            st.session_state.analysis_result = None

text_input = st.text_area(
    "Your text in Kiswahili",
    key="text",
    height=150,
    placeholder="Andika maoni yako hapa…",
    help="This tool is designed for Kiswahili text. The model may also encounter slang, spelling variation or context it does not understand.",
)

c1, c2 = st.columns([3, 1])
with c1:
    analyze_clicked = st.button("Analyze sentiment", type="primary", use_container_width=True)
with c2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.text = ""
    st.session_state.analysis_result = None
    st.rerun()

if analyze_clicked:
    if not text_input.strip():
        st.warning("Please enter a Kiswahili comment before analyzing.")
    else:
        try:
            with st.spinner("Analyzing your text…"):
                raw_results = load_model()(text_input.strip())
            if raw_results and isinstance(raw_results[0], list):
                raw_results = raw_results[0]
            raw_results = sorted(raw_results, key=lambda item: item["score"], reverse=True)
            st.session_state.analysis_result = raw_results
        except Exception as exc:
            st.error("The model could not analyze this text. Please try again.")
            st.exception(exc)

# Results remain visible until the user analyzes another sentence or clears.
results = st.session_state.analysis_result
if results:
    top = results[0]
    label = top["label"].lower()
    meta = LABEL_META.get(label, {"emoji": "🔎", "color": "#657b83", "description": "The model produced a prediction."})
    st.markdown("---")
    st.markdown("### Analysis result")
    st.markdown(f"""
    <div class="result-card" style="border-left-color:{meta['color']};">
      <div style="font-size:1.1rem;">{meta['emoji']} Predicted sentiment</div>
      <div class="result-label" style="color:{meta['color']};">{label.capitalize()}</div>
      <p style="margin:.4rem 0 0 0;color:#687b75;">{meta['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    score_cols = st.columns(3)
    for col, item in zip(score_cols, results):
        with col:
            col.markdown(f"<div class='metric-card'><div class='small-muted'>{item['label'].capitalize()} score</div><h2>{item['score']:.1%}</h2></div>", unsafe_allow_html=True)

    if top["score"] < 0.50:
        st.info("The model's top score is below 50%. Treat this prediction with extra caution.")
    elif top["score"] < 0.70:
        st.info("The model has a moderate top score. Context may affect the interpretation.")

    st.markdown("#### Model scores")
    score_data = {item["label"].capitalize(): item["score"] for item in results}
    st.bar_chart(score_data, y_label="Score", horizontal=True)
    st.caption("These are model scores, not guaranteed calibrated probabilities. The model can make mistakes.")

# Project info 
st.markdown("---")
tab_analyzer, tab_about, tab_performance, tab_limitations = st.tabs(["How to use", "About the project", "Model performance", "Limitations"])

with tab_analyzer:
    st.markdown("""
    **How to use the analyzer**
    1. Enter a Kiswahili comment, tweet or sentence.
    2. Click **Analyze sentiment**.
    3. Review the predicted class and the scores for all three classes.

    You can also select one of the example sentences above the input box.
    """)

with tab_about:
    st.markdown("""
    **Project overview**

    This project explores sentiment classification in Kiswahili text using a fine-tuned multilingual transformer model.

    - **Language:** Kiswahili (Swahili)
    - **Task:** Three-class sentiment classification
    - **Classes:** Positive, neutral and negative
    - **Dataset:** Kiswahili subset of AfriSenti-Twitter
    - **Base model:** Davlan/AfroXLMR-base
    - **Fine-tuned model:** `nafissaoued/swahili-sentiment-afroxlmr`

    The application is an academic demonstration of adapting a pretrained language model to an African-language NLP task.
    """)

with tab_performance:
    st.markdown("**Held-out test set results**")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Macro F1", "57.39%")
    metric_cols[1].metric("Accuracy", "63.0%")
    metric_cols[2].metric("Baseline Macro F1", "46.31%")
    st.markdown("**Fine-tuned model — per-class F1 scores**")
    st.bar_chart({"F1-score": {"Negative": 0.54, "Neutral": 0.72, "Positive": 0.46}}, horizontal=True)
    st.markdown("**Baseline comparison**")
    st.bar_chart({"Macro F1": {"TF-IDF + Logistic Regression": 0.4631, "Fine-tuned AfroXLMR": 0.5739}}, horizontal=True)
    st.caption("Metrics are reported from the project's held-out test evaluation.")

with tab_limitations:
    st.markdown("""
    **Known limitations**

    - The model can confuse positive and neutral text.
    - Slang, sarcasm, spelling variation and context may affect predictions.
    - The model was evaluated using a Twitter dataset; performance on other text types is not established.
    - The score shown is not a guarantee that the prediction is correct.
    - This is an academic demonstration, not a tool for high-stakes decisions.
    """)

st.markdown("---")
st.markdown("<div class='small-muted'>Built as an academic AI application exploring sentiment analysis in Kiswahili · Fine-tuned AfroXLMR</div>", unsafe_allow_html=True)
