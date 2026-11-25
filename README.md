# Emotion Recognition from Text

A Python project developed for **Advanced Programming Techniques — Project 3**.

The application predicts the emotion expressed in an English message and displays scores for six emotion classes:

**sadness, joy, love, anger, fear, surprise**

It compares two approaches:
- **TF-IDF + Logistic Regression:** a traditional machine learning baseline.
- **DistilBERT:** a transformer model fine-tuned for emotion classification.

## Team and Contributions

| Member | Main contributions |
|---|---|
| Tulin | Shared dataset preparation and cleaning, DistilBERT fine-tuning, training documentation, integration testing, and submission package preparation. |
| Shafag | Baseline model training and prediction, shared model evaluation, performance comparison, and confusion matrices. |
| Aygul | Streamlit interface, model selection and comparison views, prediction visualizations, theme adjustments, and initial setup documentation. |

The team collaborated on integration and troubleshooting.

## Application Features

- Enter an English message or select a sample message.
- Predict using either model.
- Compare both models on the same message.
- Display the predicted emotion and its prediction score.
- Visualize scores for all six emotions.
- Expand charts using the fullscreen control.
- Clear the input and results.
- Display a warning for empty input.
- Use the interface in light or dark mode.

## Dataset and Preparation

The project uses the **dair-ai/emotion** dataset from Hugging Face with its predefined `split` configuration.

The shared data loader is implemented in `src/data.py`.

| Split | Original samples | Samples after cleaning |
|---|---:|---:|
| Training | 16,000 | 15,984 |
| Validation | 2,000 | 1,997 |
| Test | 2,000 | 2,000 |

The cleaning procedure removed 16 training rows and 3 validation rows to address overlap between dataset splits. The test split was preserved.

Both models use the same emotion-label order:

| Label ID | Emotion |
|---:|---|
| 0 | sadness |
| 1 | joy |
| 2 | love |
| 3 | anger |
| 4 | fear |
| 5 | surprise |

Training data is used to fit the models, validation data supports model development and selection, and the test split is used for the final comparison.

## Models

### TF-IDF + Logistic Regression

The baseline is a scikit-learn pipeline containing:

- `TfidfVectorizer`
  - Unigrams and bigrams: `ngram_range=(1, 2)`
  - Maximum features: `100000`
  - Minimum document frequency: `2`
  - Sublinear term frequency enabled
- `LogisticRegression`
  - Maximum iterations: `1000`
  - Random seed: `42`

The saved artifact contains the trained pipeline and label names.

### Fine-Tuned DistilBERT

DistilBERT is fine-tuned for the six-class emotion classification task.

Training settings:
- Epochs: `3`
- Training batch size: `8`
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Warmup ratio: `0.1`
- Maximum token length: `128`
- Random seed: `42`
- Mixed-precision training on a CUDA GPU
- Best checkpoint selected using validation Macro F1

The predictor loads the saved model and tokenizer from `models/distilbert/`. It uses CUDA when available and otherwise runs on CPU.

Model outputs are mapped to the shared emotion-label order so that both predictors return compatible results.

## Evaluation Results

Both models were evaluated on the **same 2,000 test messages**.

| Metric | TF-IDF + Logistic Regression | DistilBERT |
|---|---:|---:|
| Accuracy | 82.95% | 92.40% |
| Macro F1 | 72.12% | 87.58% |

DistilBERT achieved higher overall accuracy and Macro F1 on this test set.

Accuracy measures the proportion of correct predictions. Macro F1 averages the F1 scores of all six classes with equal weight, making performance on less frequent emotions visible.

The evaluation script also produces:
- Per-class precision, recall, and F1 scores
- A confusion matrix for each model
- A comparison of the overall metrics

Reports and training notes are available in `reports/`.

## Requirements

**Python 3.10 is recommended.** Development and integration testing used Python 3.10.6.

Main libraries:
- PyTorch
- Hugging Face Transformers and Datasets
- scikit-learn
- pandas and NumPy
- Matplotlib and Seaborn
- Streamlit

The baseline model included in the submission package was retrained and tested with **scikit-learn 1.7.2**. Its test metrics matched the reported baseline results.

Saved scikit-learn models should be loaded using the version with which they were trained. The baseline artifact stored in GitHub may come from a different training environment; if it is incompatible, retrain it using the provided script.

## Setup

### Option 1: Submission ZIP

Extract the entire ZIP and open a terminal in the extracted folder containing `app.py`.

The submission package includes:
- The trained baseline model
- The final DistilBERT model
- The DistilBERT tokenizer files

Retraining is not required to use the application.

### Option 2: GitHub Repository

Clone the repository:

```powershell
git clone https://github.com/tulin-bayar/emotion-recognition
cd emotion-recognition
```

The DistilBERT weights are distributed separately from GitHub. Place the supplied `distilbert` folder inside `models/`.

The expected locations are:

```text
models/baseline_model.joblib
models/distilbert/config.json
models/distilbert/model.safetensors
models/distilbert/tokenizer.json
models/distilbert/tokenizer_config.json
models/distilbert/special_tokens_map.json
models/distilbert/vocab.txt
```

Avoid an extra nested directory such as `models/distilbert/distilbert/`.

### Create the Environment and Install Dependencies

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-distilbert.txt
```

For the baseline model supplied in the submission ZIP, ensure that scikit-learn 1.7.2 is installed:

```powershell
python -m pip install scikit-learn==1.7.2
```

If PowerShell blocks environment activation, allow it for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Internet access is needed to install dependencies. The application uses the supplied local model files for prediction.

The development lock files document specific environments, including a GPU environment. Use `requirements-distilbert.txt` for the installation steps above.

## Run the Application

From the folder containing `app.py`:

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal.

1. Enter an English message or select a sample.
2. Select a model, or enable **Compare both models**.
3. Click **Analyze message**.
4. Review the predicted emotion and score distribution.

Selecting a sample message fills the input box; click **Analyze message** to obtain a prediction.

If the default port is already occupied, use another port:

```powershell
python -m streamlit run app.py --server.port 8502
```

Stop the application with `Ctrl+C` in the terminal.

## Training and Evaluation

Run these commands from the project root.

Train the baseline:

```powershell
python -m src.train_baseline
```

Fine-tune DistilBERT:

```powershell
python -m src.train_distilbert
```

Evaluate both saved models:

```powershell
python -m src.evaluate
```

Training commands create or replace model artifacts. Keep a backup of existing models if you want to preserve them.

Training and evaluation require access to the dataset through Hugging Face or an existing local dataset cache. DistilBERT training is substantially more computationally demanding than baseline training.

The evaluation script displays confusion-matrix windows. Close each window to allow the script to continue.

### Using a Cached Dataset Offline

If the dataset has already been downloaded and Hugging Face is temporarily unreachable:

```powershell
$env:HF_HUB_OFFLINE = "1"
python -m src.evaluate
```

This requires an existing dataset cache; it does not download missing data.

To return to online mode:

```powershell
Remove-Item Env:HF_HUB_OFFLINE
```

## Shared Prediction Interface

Both prediction modules expose:

```python
predict_texts(texts)
predict_emotion(texts)
```

For a list of messages, they return a list of prediction dictionaries in the same order.

Each prediction contains:
- `label_id`: integer ID in the shared label order
- `label`: predicted emotion name
- `scores`: probability scores for all six emotions

Example:

```python
from src.predict_distilbert import predict_texts

results = predict_texts([
    "I am so happy today!",
    "I feel very sad."
])

for result in results:
    print(result["label"])
    print(result["scores"])
```

## Project Structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `src/data.py` | Shared dataset loading and preparation |
| `src/train_baseline.py` | Baseline training and saving |
| `src/predict_baseline.py` | Baseline prediction |
| `src/train_distilbert.py` | DistilBERT fine-tuning |
| `src/predict_distilbert.py` | DistilBERT prediction |
| `src/evaluate.py` | Shared test evaluation and confusion matrices |
| `models/` | Saved models and tokenizer files |
| `reports/` | Evaluation graphics, training notes, and integration test documentation |
| `notebooks/` | Notebook workspace |
| `requirements.txt` | Shared dependencies |
| `requirements-distilbert.txt` | Shared dependencies plus DistilBERT dependencies |

Intermediate training checkpoints and the virtual environment are excluded from the submission package.

## Integration Testing

The application was checked with:
- DistilBERT predictions
- Baseline predictions
- Side-by-side model comparison
- Sample-message selection
- Emotion-score charts
- Empty-input validation
- Clear functionality
- Light and dark interface views

The prepared submission copy was also launched locally, and both models successfully produced predictions and charts.

These checks used the existing development environment; they do not constitute a clean installation test on every supported platform.

## Limitations

- The models are intended for English text.
- There is no neutral class; every message is assigned one of the six emotions.
- Mixed emotions, sarcasm, and unfamiliar wording may lead to incorrect predictions.
- DistilBERT inputs longer than 128 tokens are truncated.
- Prediction scores are model outputs, not guaranteed probabilities of correctness.
- A high score for one message is not the model's overall test accuracy.
- Test-set performance does not guarantee the same performance on other text sources.

## References

- Dataset: https://huggingface.co/datasets/dair-ai/emotion
- Repository: https://github.com/tulin-bayar/emotion-recognition
