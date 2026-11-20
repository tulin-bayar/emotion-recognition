
"""Emotion Recognition from Text Messages — Streamlit UI."""

import matplotlib.pyplot as plt
import streamlit as st

from src.predict_distilbert import predict_texts as predict_distilbert

try:
    from src.predict_baseline import predict_texts as predict_baseline
except ImportError:
    predict_baseline = None


EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_STYLE = {
    "sadness": ("😢", "#6488C6"),
    "joy": ("😊", "#E6B84C"),
    "love": ("❤️", "#E47C9A"),
    "anger": ("😠", "#DF6B61"),
    "fear": ("😨", "#9278C8"),
    "surprise": ("😲", "#55AFA8"),
}

SAMPLES = [
    "I feel so happy and excited about my future!",
    "I miss my family and feel very lonely today.",
    "I am terrified of what might happen next.",
    "I love spending time with the people I care about.",
]


st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="💬",
    layout="wide",
)


# -------------------- DESIGN --------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F7F7FC;
        color: #283451;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #283451 !important;
        font-weight: 700 !important;
    }

    h1 {
        font-size: 2rem !important;
    }

    .subtitle {
        color: #78839A;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .result-card {
        background: #FFFFFF;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #EAE6FA;
        text-align: center;
        box-shadow: 0 3px 12px rgba(40, 52, 81, 0.04);
        margin-bottom: 15px;
    }

    .emotion-name {
        font-size: 2rem;
        font-weight: 750;
        margin: 8px 0;
    }

    .muted {
        color: #78839A;
        font-size: 0.9rem;
    }

    div.stButton > button {
        border-radius: 11px;
        border: 1px solid #DED9F2;
        color: #51477D;
        background-color: #FFFFFF;
        font-weight: 600;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #8172CC;
        color: #8172CC;
        background-color: #F1EEFC;
    }

    div.stButton > button[kind="primary"] {
        background-color: #8172CC;
        border-color: #8172CC;
        color: white;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #6E5DB9;
        border-color: #6E5DB9;
        color: white;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        border-radius: 11px;
        border-color: #DED9F2;
    }

    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #EAE6FA;
        border-radius: 12px;
    }

    hr {
        border-color: #EAE6FA;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- CHART --------------------

def draw_chart(scores, dominant):
    values = [scores.get(e, 0.0) * 100 for e in EMOTIONS]

    colors = [
        EMOTION_STYLE[e][1] if e == dominant else "#E5E1F2"
        for e in EMOTIONS
    ]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(
        EMOTIONS,
        values,
        color=colors,
        height=0.58
    )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Prediction score (%)", color="#78839A")
    ax.invert_yaxis()

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#EAE6FA")

    ax.tick_params(axis="y", length=0, colors="#283451")
    ax.tick_params(axis="x", colors="#78839A")
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
            color="#283451",
        )

    fig.tight_layout()
    return fig


# -------------------- RESULT CARD --------------------

def show_result(result, model_name):
    label = result["label"].lower()
    scores = result["scores"]

    emoji, color = EMOTION_STYLE.get(
        label,
        ("💬", "#8172CC")
    )

    confidence = scores.get(label, 0.0)

    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size: 2.6rem;">{emoji}</div>
            <div class="muted">PREDICTED EMOTION</div>
            <div class="emotion-name" style="color:{color};">
                {label.capitalize()}
            </div>
            <div class="muted">{model_name}</div>
            <h2 style="color:#283451;">{confidence:.1%}</h2>
            <div class="muted">Prediction score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "The prediction score represents the model's confidence "
        "for this message. It is not the model's overall accuracy."
    )

    st.markdown("#### Emotion distribution")

    fig = draw_chart(scores, label)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# -------------------- SESSION STATE --------------------

if "message" not in st.session_state:
    st.session_state.message = ""

if "results" not in st.session_state:
    st.session_state.results = None


def load_sample(sample):
    st.session_state.message = sample


def clear_input():
    st.session_state.message = ""
    st.session_state.results = None


# -------------------- HEADER --------------------

st.title("💬 Emotion Recognition")

st.markdown(
    '<p class="subtitle">'
    'Discover the emotions behind your text messages '
    'using machine learning.'
    '</p>',
    unsafe_allow_html=True,
)


# -------------------- MAIN LAYOUT --------------------

left, right = st.columns([1, 1.15], gap="large")


with left:
    st.markdown("### ✍️ Your message")

    st.text_area(
        "Enter an English message",
        key="message",
        height=170,
        max_chars=1000,
        placeholder="Write something here... How are you feeling today?",
        label_visibility="collapsed",
    )

    st.caption(
        f"{len(st.session_state.message)} / 1000 characters"
    )

    st.markdown("**Try a sample message:**")

    for i, sample in enumerate(SAMPLES):
        st.button(
            sample,
            key=f"sample_{i}",
            use_container_width=True,
            on_click=load_sample,
            args=(sample,),
        )

    model_options = ["DistilBERT"]

    if predict_baseline is not None:
        model_options.append("TF-IDF + Logistic Regression")

    model_name = st.selectbox(
        "Select model",
        model_options,
    )

    compare = st.checkbox(
        "Compare both models",
        disabled=predict_baseline is None,
        help=(
            "Available when the baseline prediction module "
            "is connected."
        ),
    )

    analyze_col, clear_col = st.columns(2)

    with analyze_col:
        analyze = st.button(
            "🔍 Analyze message",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        st.button(
            "Clear",
            use_container_width=True,
            on_click=clear_input,
        )

    if analyze:
        message = st.session_state.message.strip()

        if not message:
            st.warning("Please enter a message before analyzing.")

        else:
            try:
                with st.spinner("Analyzing your message..."):

                    results = {}

                    if compare:
                        results["DistilBERT"] = (
                            predict_distilbert([message])[0]
                        )

                        results["TF-IDF + Logistic Regression"] = (
                            predict_baseline([message])[0]
                        )

                    elif model_name == "DistilBERT":
                        results["DistilBERT"] = (
                            predict_distilbert([message])[0]
                        )

                    else:
                        results[model_name] = (
                            predict_baseline([message])[0]
                        )

                    st.session_state.results = results

            except FileNotFoundError:
                st.error(
                    "A required model file is missing. "
                    "Please check the models folder and model setup."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# -------------------- RESULTS --------------------

with right:
    st.markdown("### 📊 Analysis results")

    results = st.session_state.results

    if results:

        if len(results) == 2:
            result_cols = st.columns(2)

            for col, (name, result) in zip(
                result_cols,
                results.items()
            ):
                with col:
                    st.markdown(f"**{name}**")
                    show_result(result, name)

        else:
            name, result = next(iter(results.items()))
            show_result(result, name)

    else:
        st.info(
            "Your prediction results will appear here. "
            "Enter a message and click Analyze."
        )


# -------------------- ABOUT PROJECT --------------------

st.divider()

with st.expander("ℹ️ About the project"):

    st.markdown("""
    ### Emotion Recognition from Text Messages

    This project classifies English text messages into six emotion categories.

    **Supported emotions:**

    - 😢 Sadness
    - 😊 Joy
    - ❤️ Love
    - 😠 Anger
    - 😨 Fear
    - 😲 Surprise

    **Machine learning models:**

    - **DistilBERT:** Transformer-based text classification.
    - **TF-IDF + Logistic Regression:** Traditional machine learning baseline.

    **Project information:**

    - Input language: English
    - Number of emotion classes: 6
    - Neutral emotion: Not included

    The models predict one of the six supported emotions for each message.
    """)
