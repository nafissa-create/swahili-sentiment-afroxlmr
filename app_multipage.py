import streamlit as st
import pandas as pd
from transformers import pipeline

MODEL_ID = "nafissaoued/swahili-sentiment-afroxlmr"
st.set_page_config(page_title="Kiswahili Sentiment AI", page_icon="💬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background: #f5f8f6; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1180px; }
.hero { background: linear-gradient(125deg, #102f2c 0%, #1b5146 70%, #23785d 100%); padding: 2.5rem 2.2rem; border-radius: 24px; color: white; margin-bottom: 1.5rem; box-shadow: 0 12px 30px rgba(16,47,44,.10); }
.hero-kicker { color: #a9ddc6; letter-spacing: .13em; font-weight: 700; font-size: .78rem; text-transform: uppercase; }
.hero h1 { color: white; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: clamp(2rem, 5vw, 3.15rem); line-height: 1.12; margin: .55rem 0; }
.hero p { color: #d4e5df; font-size: 1.03rem; max-width: 760px; margin-bottom: 0; }
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

def hero(kicker, title, description):
    st.markdown(f"""<div class="hero"><div class="hero-kicker">{kicker}</div><h1>{title}</h1><p>{description}</p></div>""", unsafe_allow_html=True)

def footer():
    st.markdown("---")
    st.markdown("<p class='small-muted' style='text-align:center;'>Kiswahili Sentiment AI · Academic NLP project · Fine-tuned AfroXLMR</p>", unsafe_allow_html=True)

# -------------------- Page 1: Home --------------------
def home():
    hero("Kiswahili AI · NLP Mini-project", "Kiswahili Sentiment AI", "Exploring how artificial intelligence can interpret sentiment expressed in Kiswahili text.")
    st.markdown("### Can AI understand the sentiment behind Kiswahili text?")
    st.markdown("This project fine-tunes a multilingual transformer model to classify Kiswahili text into **positive, neutral or negative** sentiment. Explore the project, try the analyzer and examine its evaluation results.")
    a, b, c = st.columns(3)
    a.metric("Sentiment classes", "3")
    b.metric("Language", "Kiswahili")
    c.metric("Model", "AfroXLMR")
    st.markdown("### Explore the project")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### 📘 Project overview")
        st.write("Learn about the problem, dataset, approach and model development.")
    with cols[1]:
        st.markdown("#### 💬 Try the analyzer")
        st.write("Enter a Kiswahili comment and see how the model classifies it.")
    with cols[2]:
        st.markdown("#### 📊 Evaluation")
        st.write("Explore performance metrics, class-level results and the confusion matrix.")
    st.info("Use the navigation menu in the sidebar to explore each section.")
    footer()

# -------------------- Page 2: About --------------------
def about():
    hero("Project background", "About the project", "A small NLP project exploring sentiment classification for Kiswahili text.")
    st.markdown("## The problem and motivation")
    st.write("Sentiment analysis is a natural language processing task that identifies the sentiment expressed in text. While many language technologies are developed for widely represented languages, resources and tools for African languages remain comparatively limited. This project explores sentiment classification in Kiswahili text.")
    st.markdown("## Project overview")
    x, y, z = st.columns(3)
    x.markdown("#### 🌍 Language\n**Kiswahili**\n\nA Swahili-language NLP application.")
    y.markdown("#### 🧠 Task\n**3-class classification**\n\nPositive · Neutral · Negative")
    z.markdown("#### 📚 Dataset\n**AfriSenti-Twitter**\n\nKiswahili subset")
    st.markdown("## Dataset")
    st.write("The project uses the Kiswahili subset of the AfriSenti-Twitter dataset, which contains tweets annotated with sentiment labels. The data were prepared for a three-class classification task.")
    st.markdown("## Model and approach")
    st.write("A pretrained multilingual transformer was fine-tuned on labeled Kiswahili data. Fine-tuning adapts the model's learned language representations to the sentiment classification task.")
    st.markdown("### AI pipeline")
    p = st.columns([1,.25,1,.25,1,.25,1])
    p[0].markdown("**1 · Input**\n\nKiswahili text")
    p[1].markdown("<div style='text-align:center;padding-top:1.2rem;font-size:1.5rem'>→</div>",unsafe_allow_html=True)
    p[2].markdown("**2 · Base model**\n\nAfroXLMR")
    p[3].markdown("<div style='text-align:center;padding-top:1.2rem;font-size:1.5rem'>→</div>",unsafe_allow_html=True)
    p[4].markdown("**3 · Fine-tuning**\n\nLabeled data")
    p[5].markdown("<div style='text-align:center;padding-top:1.2rem;font-size:1.5rem'>→</div>",unsafe_allow_html=True)
    p[6].markdown("**4 · Output**\n\n3 sentiment labels")
    st.markdown("### Model details")
    st.markdown("- **Base model:** `Davlan/afro-xlmr-base`\n- **Architecture:** multilingual transformer\n- **Fine-tuned model:** `nafissaoued/swahili-sentiment-afroxlmr`\n- **Application:** Streamlit and Transformers")
    footer()

# -------------------- Page 3: Analyzer --------------------
def analyzer():
    hero("Interactive demonstration", "Sentiment analyzer", "Try a sample or enter your own Kiswahili text to explore the model's predictions.")
    st.markdown("<p class='section-intro'>Choose an example or enter your own Kiswahili text to get started.</p>", unsafe_allow_html=True)
    if "text" not in st.session_state: st.session_state.text = ""
    if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
    st.markdown("<div class='example-wrap'><strong>Try an example</strong><p class='small-muted'>Select a sample sentence to fill the text box.</p></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    icons = {"Positive":"😊", "Negative":"☹️", "Neutral":"😐"}
    for col, (label, sentence) in zip(cols, EXAMPLES.items()):
        with col:
            if st.button(f"{icons[label]}  {label}", use_container_width=True, key=f"example_{label}"):
                st.session_state.text = sentence
                st.session_state.analysis_result = None
    text_input = st.text_area("Your comment", key="text", height=140, placeholder="Andika maoni yako hapa…", help="This tool is designed for Kiswahili text. Slang, spelling variation and context may affect the prediction.")
    c1, c2 = st.columns([3,1])
    with c1: analyze_clicked = st.button("🔎  Analyze sentiment", type="primary", use_container_width=True)
    with c2: clear_clicked = st.button("Clear", use_container_width=True)
    if clear_clicked:
        st.session_state.text = ""; st.session_state.analysis_result = None; st.rerun()
    if analyze_clicked:
        if not text_input.strip(): st.warning("Please enter a Kiswahili comment before analyzing.")
        else:
            try:
                with st.spinner("Analyzing your text…"):
                    raw_results = load_model()(text_input.strip())
                if raw_results and isinstance(raw_results[0], list): raw_results = raw_results[0]
                st.session_state.analysis_result = sorted(raw_results, key=lambda item: item["score"], reverse=True)
            except Exception:
                st.error("The model could not analyze this text. Please try again.")
    results = st.session_state.analysis_result
    if results:
        top = results[0]; label = top["label"].lower()
        meta = LABEL_META.get(label, {"emoji":"🔎", "color":"#657b83", "description":"The model produced a prediction."})
        st.markdown("---\n### Analysis result")
        st.markdown(f"<div class='result-card' style='border-left-color:{meta['color']};'><div style='font-size:1rem;color:#687b75;'>{meta['emoji']} Predicted sentiment</div><div style='font-family:Manrope,sans-serif;font-size:2.2rem;font-weight:800;color:{meta['color']};margin:.2rem 0;'>{label.capitalize()}</div><p style='margin:.2rem 0 0 0;color:#687b75;'>{meta['description']}</p></div>", unsafe_allow_html=True)
        score_lookup = {item["label"].lower(): item["score"] for item in results}; ordered_labels = ["negative","neutral","positive"]
        score_cols = st.columns(3)
        for col, name in zip(score_cols, ordered_labels):
            with col: col.markdown(f"<div class='metric-card'><div class='small-muted'>{name.capitalize()}</div><h2>{score_lookup.get(name,0):.1%}</h2></div>", unsafe_allow_html=True)
        if top["score"] < .50: st.warning("The model's top score is below 50%. Treat this prediction with extra caution.")
        elif top["score"] < .70: st.info("The model has a moderate top score. Context may affect the interpretation.")
        st.markdown("#### Sentiment score distribution")
        st.bar_chart({name.capitalize():score_lookup.get(name,0) for name in ordered_labels}, y_label="Model score", horizontal=True)
        st.caption("Scores represent the model's outputs and are not guaranteed to be calibrated probabilities.")
    footer()

# -------------------- Page 4: Evaluation --------------------
def evaluation():
    hero("Testing and results", "Model evaluation", "A closer look at how the fine-tuned model performed on the held-out test set.")
    st.markdown("## Overall performance")
    m = st.columns(3)
    m[0].metric("Macro F1", "57.39%")
    m[1].metric("Accuracy", "63.0%")
    m[2].metric("Baseline Macro F1", "46.31%")
    st.markdown("## Comparison with baseline")
    chart = st.columns(2)
    with chart[0]:
        st.markdown("#### Macro F1 comparison")
        st.bar_chart({"Macro F1":{"TF-IDF + Logistic Regression":.4631,"Fine-tuned AfroXLMR":.5739}}, horizontal=True)
    with chart[1]:
        st.markdown("#### Fine-tuned model · F1 by class")
        st.bar_chart({"F1-score":{"Negative":.54,"Neutral":.72,"Positive":.46}}, horizontal=True)
    st.markdown("## Confusion matrix")
    st.caption("Rows show the actual class; columns show the model's prediction.")
    confusion = [[43,32,5],[29,342,73],[6,129,89]]
    confusion_df = pd.DataFrame(confusion, index=["Actual Negative","Actual Neutral","Actual Positive"], columns=["Predicted Negative","Predicted Neutral","Predicted Positive"])
    st.dataframe(confusion_df, use_container_width=True)
    st.markdown("### Interpretation")
    st.write("The model performs best on neutral text. Positive examples are often classified as neutral, illustrating the difficulty of interpreting implicit sentiment, context and informal language.")
    st.markdown("## Limitations and responsible use")
    st.markdown("- The model can confuse positive and neutral text.\n- Slang, sarcasm, spelling variation and context may affect predictions.\n- It was evaluated using a Twitter dataset; performance on other text types is not established.\n- Model scores are not guarantees of correctness or calibrated probabilities.\n- This is an academic demonstration, not a tool for high-stakes decisions.")
    footer()

pages = {
    "Home": st.Page(home, title="Home", icon="🏠", default=True),
    "About the Project": st.Page(about, title="About the Project", icon="📘"),
    "Sentiment Analyzer": st.Page(analyzer, title="Sentiment Analyzer", icon="💬"),
    "Model Evaluation": st.Page(evaluation, title="Model Evaluation", icon="📊"),
}
pg = st.navigation(pages, position="sidebar")
pg.run()
