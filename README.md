# Emotion Recognition from Text

Text emotion recognition project for Advanced Programming Techniques.
Predicts the emotion of an English message (sadness, joy, love, anger, fear, surprise)
and shows the scores in a chart. Two models are compared: TF-IDF + Logistic Regression
and a fine-tuned DistilBERT.

## Setup

Python 3.10 is recommended (3.10.6 was used).

```bash
git clone https://github.com/tulin-bayar/emotion-recognition
cd emotion-recognition
py -3.10 -m venv .venv          # Windows
.venv\Scripts\activate
pip install -r requirements-distilbert.txt
```

Do not install `requirements-distilbert-lock.txt` (it belongs to a GPU environment).

## Model files

Model files are too large for GitHub and are shared separately. Unzip the shared archive
and place the `distilbert` folder here:

```
emotion-recognition/models/distilbert/
    config.json, model.safetensors, tokenizer.json, vocab.txt, ...
```

The folder must not be nested (`models/distilbert/distilbert/` is wrong).

## Run

```bash
streamlit run app.py
```

Type a message, choose a model, and press **Analyze**. The app shows the dominant
emotion and a bar chart of all six scores. DistilBERT runs on CPU if no GPU is available.

## Project structure

- `app.py` - Streamlit interface
- `src/predict_distilbert.py` - DistilBERT predictor
- `src/predict_baseline.py` - TF-IDF + Logistic Regression predictor
- `src/data.py` - shared data loading
- `models/` - local model files (not committed)
