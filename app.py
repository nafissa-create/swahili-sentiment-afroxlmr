import streamlit as st
from transformers import pipeline

MODEL_ID = "nafissaoued/swahili-sentiment-afroxlmr"

st.set_page_config(page_title="Kiswahili Sentiment Analyzer", page_icon="💬")


@st.cache_resource(show_spinner="Loading the model (the first time can take a few minutes)...")
def load_model():
    return pipeline("text-classification", model=MODEL_ID, top_k=None)


st.title("Kiswahili Sentiment Analyzer")
st.write(
    "Paste a Kiswahili comment or tweet. The model estimates whether it is "
    "**positive**, **negative** or **neutral**."
)

EXAMPLES = {
    "Positive": "Huduma ilikuwa nzuri sana, asante!",
    "Negative": "Hii ni mbaya sana, sijafurahi",
    "Neutral": "Mkutano utafanyika kesho saa tatu",
}

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (name, sentence) in zip(cols, EXAMPLES.items()):
    if col.button(name):
        st.session_state["text"] = sentence

text = st.text_area("Comment", key="text", height=120)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please type or paste a comment first.")
    else:
        results = load_model()(text)
        if isinstance(results[0], list):   # some library versions return a nested list
            results = results[0]
        results = sorted(results, key=lambda r: -r["score"])

        top = results[0]
        st.subheader(f"Prediction: {top['label']} ({top['score']:.0%})")
        if top["score"] < 0.5:
            st.info("The model is not very sure about this one: the top score is below 50%.")

        st.bar_chart({r["label"]: r["score"] for r in results})

st.divider()
st.caption(
    "Model: AfroXLMR-base fine-tuned on the Kiswahili part of AfriSenti (tweets). "
    "It can make mistakes, especially between neutral and positive, and it has not been "
    "tested on other kinds of text such as TikTok comments. For academic, non-commercial use."
)
