"""Streamlit UI for emotion recognition (run with: streamlit run app.py)."""
import matplotlib.pyplot as plt
import streamlit as st

from src.predict_distilbert import predict_texts as predict_distilbert

try:
    from src.predict_baseline import predict_texts as predict_baseline
except ImportError:
    predict_baseline = None

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MODELS = {
    "DistilBERT": predict_distilbert,
    "TF-IDF + Logistic Regression": predict_baseline,
}

st.set_page_config(page_title="Emotion Recognition", page_icon="🙂")
st.title("Emotion Recognition from Text")
st.write("Type an English message and the model will predict its emotion.")

model_name = st.radio("Model", list(MODELS.keys()), horizontal=True)
text = st.text_area(
    "Your message",
    height=140,
    placeholder="e.g. I can't believe how wonderful this day turned out!",
)


def draw_chart(scores, dominant):
    values = [scores.get(e, 0.0) for e in EMOTIONS]
    colors = ["#d62728" if e == dominant else "#9aa5b1" for e in EMOTIONS]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(EMOTIONS, values, color=colors)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Emotion scores")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.1%}",
                ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter a message first.")
    elif MODELS[model_name] is None:
        st.error(f"{model_name} is not available yet (src/predict_baseline.py not found).")
    else:
        try:
            with st.spinner("Analyzing..."):
                result = MODELS[model_name]([text])[0]
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            st.subheader(f"Dominant emotion: {result['label']}")
            st.caption(f"Confidence: {result['scores'][result['label']]:.1%} ({model_name})")
            fig = draw_chart(result["scores"], result["label"])
            st.pyplot(fig)
            plt.close(fig)
