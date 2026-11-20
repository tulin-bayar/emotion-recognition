# DistilBERT Training — Tülin Bayar

## My Contribution

I prepared the shared project structure and Python environment,
explored the emotion dataset, checked data quality, and implemented
the shared data-loading function.

I then fine-tuned DistilBERT, saved the model and tokenizer,
verified that the saved model could run on CPU, and shared
the model files with the team.

## Dataset and Preparation

We used the English dair-ai/emotion dataset with six labels:

- 0: sadness
- 1: joy
- 2: love
- 3: anger
- 4: fear
- 5: surprise

The original training set contained no empty messages.
We found 31 duplicate text occurrences and 30 distinct texts
with conflicting emotion labels. We preserved the original labels.

To prevent exact text overlap between dataset splits, we kept
the test set unchanged and removed overlapping messages from
the training and validation sets.

Final split sizes:

- Training: 15,984 messages
- Validation: 1,997 messages
- Test: 2,000 messages

Both models must use the shared load_emotion_data() function
in src/data.py.

## Model and Training Settings

Base model: distilbert/distilbert-base-uncased

The pretrained model was fine-tuned for six-class emotion
classification using the following settings:

- Epochs: 3
- Training batch size: 8
- Evaluation batch size: 8
- Learning rate: 0.00002
- Maximum input length: 128 tokens
- Weight decay: 0.01
- Warmup ratio: 0.1
- Random seed: 42
- Mixed precision: FP16
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM

A short 20-step training trial was completed before full training.
Full training started again from the pretrained base model.

## Model Selection and Validation Results

The model was evaluated after each epoch.
The checkpoint with the highest validation macro F1 was selected.

- Validation accuracy: 0.9429 (94.29%)
- Validation macro F1: 0.9194
- Selected checkpoint: checkpoint-3996

Macro F1 gives equal weight to each emotion class, which is
important because the dataset has an imbalanced class distribution.

These are validation results, not final test results.
The test set was not used for training or model selection.

## Saved Model and Handoff

The selected model and tokenizer were saved in:

models/distilbert/

The folder was shared with the team as distilbert-model.zip.
Large model files are excluded from Git and are shared separately.

The saved model was successfully loaded on CPU.
For the example message:

"I am so happy to see my friends again!"

the model predicted joy.

This single example is a loading and inference check,
not a measurement of overall model performance.

For inference, use truncation=True and max_length=128.

## Environment

- Python: 3.10.6
- PyTorch: 2.8.0+cu126
- Transformers: 4.57.1
- Datasets: 4.3.0
- Hugging Face Hub: 0.36.2
- Accelerate: 1.15.0

Training command, run from the project root:

python -m src.train_distilbert

## Limitations and Remaining Work

The model was trained on English messages.
It does not include a neutral class and may assign neutral
messages to one of the six available emotions.

Long inputs are truncated to 128 tokens.
Model scores should not be interpreted as guaranteed correctness.

Remaining team tasks include application integration,
final test evaluation, and comparison with the baseline model.