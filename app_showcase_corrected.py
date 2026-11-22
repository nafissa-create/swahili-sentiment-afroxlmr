import streamlit as st
import pandas as pd
from transformers import pipeline

MODEL_ID = "nafissaoued/swahili-sentiment-afroxlmr"

st.set_page_config(page_title="Kiswahili Sentiment AI", page_icon="💬", layout="wide")

# ----------------------------- Styling --------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background: #f5f8f6; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1180px; }
.hero { background: linear-gradient(125deg, #102f2c 0%, #1b5146 70%, #23785d 100%); padding: 2.25rem 2.2rem; border-radius: 24px; color: white; margin-bottom: 1.5rem; box-shadow: 0 12px 30px rgba(16,47,44,.10); }
.hero-kicker { color: #a9ddc6; letter-spacing: .13em; font-weight: 700; font-size: .78rem; text-transform: uppercase; }
.hero h1 { color: white; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: clamp(2rem, 5vw, 3.15rem); line-height: 1.12; margin: .55rem 0; }
.hero p { color: #d4e5df; font-size: 1.03rem; max-width: 680px; margin-bottom: 0; }
.section-intro { color: #61746d; margin-top: -.5rem; margin-bottom: 1.2rem; }
.example-wrap { background: white; border: 1px solid #e4ece7; border-radius: 18px; padding: 1.1rem 1.2rem .8rem; margin-bottom: 1rem; }
.result-card { background: white; border: 1px solid #e5ece8; border-left: 6px solid #23785d; border-radius: 18px; padding: 1.4rem 1.5rem; }
.metric-card { background: white; border: 1px solid #e5ece8; border-radius: 16px; padding: 1rem; }
.small-muted { color: #687b75; font-size: .92rem; }
div.stButton > button[kind="primary"] { background: #1c6852; border: 1px solid #1c6852; border-radius: 12px; font-weight: 700; min-height: 3rem; }
div.stButton > button { border-radius: 12px; min-height: 2.8rem; }
[data-testid="stTextArea"] textarea { border-radius: 14px; }
hr { border-color: #e4ebe7; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Model ---------------------------------------
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

# ----------------------------- Header --------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Kiswahili AI · NLP Project</div>
  <h1>Understand the sentiment<br>behind Kiswahili text.</h1>
  <p>Explore how AI interprets sentiment in Kiswahili comments, messages and tweets.<br>Classify text as positive, negative or neutral.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Analyzer ------------------------------------
st.markdown("### Sentiment analyzer")
st.markdown("<p class='section-intro'>Choose an example or enter your own Kiswahili text to get started.</p>", unsafe_allow_html=True)

if "text" not in st.session_state:
    st.session_state.text = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

st.markdown("<div class='example-wrap'><strong>Try an example</strong><p class='small-muted'>Select a sample sentence to automatically fill the text box.</p></div>", unsafe_allow_html=True)
cols = st.columns(3)
example_icons = {"Positive": "😊", "Negative": "☹️", "Neutral": "😐"}
for col, (label, sentence) in zip(cols, EXAMPLES.items()):
    with col:
        if st.button(f"{example_icons[label]}  {label}", use_container_width=True, key=f"example_{label}"):
            st.session_state.text = sentence
            st.session_state.analysis_result = None

text_input = st.text_area(
    "Your comment",
    key="text",
    height=140,
    placeholder="Andika maoni yako hapa…",
    help="This tool is designed for Kiswahili text. Slang, spelling variation and context may affect the prediction.",
)

c1, c2 = st.columns([3, 1])
with c1:
    analyze_clicked = st.button("🔎  Analyze sentiment", type="primary", use_container_width=True)
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
            st.session_state.analysis_result = sorted(raw_results, key=lambda item: item["score"], reverse=True)
        except Exception:
            st.error("The model could not analyze this text. Please try again.")

# ----------------------------- Results -------------------------------------
results = st.session_state.analysis_result
if results:
    top = results[0]
    label = top["label"].lower()
    meta = LABEL_META.get(label, {"emoji": "🔎", "color": "#657b83", "description": "The model produced a prediction."})
    st.markdown("---")
    st.markdown("### Analysis result")
    st.markdown(f"""
    <div class="result-card" style="border-left-color:{meta['color']};">
      <div style="font-size:1rem;color:#687b75;">{meta['emoji']} Predicted sentiment</div>
      <div style="font-family:Manrope,sans-serif;font-size:2.2rem;font-weight:800;color:{meta['color']};margin:.2rem 0;">{label.capitalize()}</div>
      <p style="margin:.2rem 0 0 0;color:#687b75;">{meta['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    score_lookup = {item["label"].lower(): item["score"] for item in results}
    ordered_labels = ["negative", "neutral", "positive"]
    score_cols = st.columns(3)
    for col, name in zip(score_cols, ordered_labels):
        with col:
            score = score_lookup.get(name, 0)
            col.markdown(f"<div class='metric-card'><div class='small-muted'>{name.capitalize()}</div><h2>{score:.1%}</h2></div>", unsafe_allow_html=True)

    if top["score"] < 0.50:
        st.warning("The model's top score is below 50%. Treat this prediction with extra caution.")
    elif top["score"] < 0.70:
        st.info("The model has a moderate top score. Context may affect the interpretation.")

    st.markdown("#### Sentiment score distribution")
    st.bar_chart({name.capitalize(): score_lookup.get(name, 0) for name in ordered_labels}, y_label="Model score", horizontal=True)
    st.caption("Scores represent the model's outputs and are not guaranteed to be calibrated probabilities.")

# ----------------------------- Project showcase -------------------------------
st.markdown("---")
st.markdown("## About the project")
st.markdown("""
This project explores how a multilingual AI model can identify sentiment in **Kiswahili text**.
Sentiment analysis can help summarize opinions expressed in comments, messages and social-media text.
The project adapts a pretrained transformer to classify text into **positive, neutral or negative** sentiment.
""")

# Three overview cards
intro_cols = st.columns(3)
with intro_cols[0]:
    st.markdown("#### 🌍 Language")
    st.markdown("**Kiswahili**")
    st.caption("An African-language NLP application")
with intro_cols[1]:
    st.markdown("#### 🧠 Task")
    st.markdown("**3-class classification**")
    st.caption("Positive · Neutral · Negative")
with intro_cols[2]:
    st.markdown("#### 📚 Dataset")
    st.markdown("**AfriSenti-Twitter**")
    st.caption("Kiswahili subset")

st.markdown("## How the AI works")
st.markdown("The model receives a Kiswahili text input, processes it using a fine-tuned multilingual transformer, and returns scores for three sentiment classes.")
pipeline_cols = st.columns([1, .25, 1, .25, 1, .25, 1])
with pipeline_cols[0]:
    st.markdown("**1 · Input**")
    st.caption("Kiswahili comment or tweet")
with pipeline_cols[1]:
    st.markdown("<div style='text-align:center;padding-top:1.2rem;font-size:1.5rem'>→</div>", unsafe_allow_html=True)
with pipeline_cols[2]:
    st.markdown("**2 · Base model**")
    st.caption("Davlan / AfroXLMR-base")
with pipeline_cols[3]:
    st.markdown("<div style='text-align:center;padding-top:1.2rem;font-size:1.5rem'>→</div>", unsafe_allow_html=True)
with pipeline_cols[4]:
    st.markdown("**3 · Fine-tuning**")
    st.caption("Adapted using labeled Kiswahili data")
with pipeline_cols[5]:
    st.markdown("<div style='text-align:center;padding-top:1.2rem;font-size:1.5rem'>→</div>", unsafe_allow_html=True)
with pipeline_cols[6]:
    st.markdown("**4 · Prediction**")
    st.caption("Positive · Neutral · Negative")

st.markdown("## Model performance")
st.markdown("Results below are from the project's held-out test set.")
perf_cols = st.columns(3)
perf_cols[0].metric("Macro F1", "57.39%")
perf_cols[1].metric("Accuracy", "63.0%")
perf_cols[2].metric("Baseline Macro F1", "46.31%")

chart_cols = st.columns(2)
with chart_cols[0]:
    st.markdown("#### Macro F1 comparison")
    st.bar_chart({"Macro F1": {"TF-IDF + Logistic Regression": 0.4631, "Fine-tuned AfroXLMR": 0.5739}}, horizontal=True)
with chart_cols[1]:
    st.markdown("#### Fine-tuned model · F1 by class")
    st.bar_chart({"F1-score": {"Negative": 0.54, "Neutral": 0.72, "Positive": 0.46}}, horizontal=True)

st.markdown("#### Confusion matrix")
st.caption("Rows show the actual class; columns show the model's prediction.")
confusion = {
    "Predicted Negative": [43, 29, 6],
    "Predicted Neutral": [32, 342, 129],
    "Predicted Positive": [5, 73, 89],
}
confusion_df = pd.DataFrame(confusion, index=["Actual Negative", "Actual Neutral", "Actual Positive"])
st.dataframe(confusion_df, use_container_width=True)
st.markdown("**What we observed:** the model performs best on neutral text. Positive text is often classified as neutral, highlighting the difficulty of interpreting implicit sentiment, context and informal language.")

st.markdown("## Dataset and model")
info_cols = st.columns(2)
with info_cols[0]:
    st.markdown("#### Dataset")
    st.markdown("""
- **Source:** AfriSenti-Twitter
- **Language subset:** Kiswahili
- **Task labels:** negative, neutral and positive
- **Data preparation:** inspected duplicates and split overlap, then prepared training and validation data.
""")
with info_cols[1]:
    st.markdown("#### Model")
    st.markdown("""
- **Base model:** `Davlan/afro-xlmr-base`
- **Architecture:** multilingual transformer
- **Fine-tuned model:** `nafissaoued/swahili-sentiment-afroxlmr`
- **Application:** Streamlit + Transformers
""")

st.markdown("## Limitations and responsible use")
st.markdown("""
- The model can confuse positive and neutral text.
- Slang, sarcasm, spelling variation and context may affect predictions.
- It was evaluated using a Twitter dataset; performance on other text types is not established.
- Model scores are not guarantees of correctness or calibrated probabilities.
- This is an academic demonstration, not a tool for high-stakes decisions.
""")

st.markdown("---")
st.markdown("<p class='small-muted' style='text-align:center;'>Kiswahili Sentiment Analyzer · Academic AI application · Fine-tuned AfroXLMR</p>", unsafe_allow_html=True)
