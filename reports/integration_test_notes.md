# Integration Test Notes — Tülin Bayar

Date: 2026-11-20

## Environment
- Python 3.10.6
- scikit-learn 1.7.2
- PyTorch 2.8.0+cu126
- NVIDIA GeForce RTX 4070 Laptop GPU

## Model Evaluation
Both models were evaluated on the same 2,000 test samples.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 82.95% | 72.12% |
| DistilBERT | 92.40% | 87.58% |

The baseline was retrained locally with scikit-learn 1.7.2.
Its test metrics matched the teammate's reported results
at the displayed precision. DistilBERT was not retrained.

## UI Checks
- DistilBERT displayed predictions and emotion scores.
- Compare both models displayed both predictions and charts.
- Both models predicted Fear for:
  "I am terrified of what might happen next."
- Empty input displayed:
  "Please enter a message before analyzing."

## Local Model
The locally retrained baseline is backed up outside the repository.
It has not been committed over the shared model.