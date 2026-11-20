import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import load_emotion_data


def compute_metrics(prediction):
    # Select the emotion with the highest model score.
    # (En yüksek model skoruna sahip duyguyu seç.)
    predictions = np.argmax(prediction.predictions, axis=-1)
    labels = prediction.label_ids

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            labels=list(range(6)),
            average="macro",
            zero_division=0,
        ),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not detected.")

    print("GPU:", torch.cuda.get_device_name(0))
    set_seed(42)

    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models" / "distilbert"
    checkpoint_dir = project_root / "models" / "distilbert-checkpoints"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Use all training and validation examples.
    # (Tüm eğitim ve doğrulama örneklerini kullan.)
    train_df, validation_df, _, label_names = load_emotion_data()

    train_dataset = Dataset.from_pandas(
        train_df, preserve_index=False
    )
    validation_dataset = Dataset.from_pandas(
        validation_df, preserve_index=False
    )

    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(
        tokenize, batched=True, remove_columns=["text"]
    )
    validation_dataset = validation_dataset.map(
        tokenize, batched=True, remove_columns=["text"]
    )

    id2label = dict(enumerate(label_names))
    label2id = {name: number for number, name in id2label.items()}

    # Start from the pretrained model, not the short trial.
    # (Kısa denemeden değil, önceden eğitilmiş temel modelden başla.)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate and save the best validation model.
    # (Doğrulamada en iyi olan modeli değerlendir ve kaydet.)
    results = trainer.evaluate()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Keep the inference length consistent with training.
    # (Tahmin aşamasındaki uzunluğu eğitimle tutarlı tut.)
    tokenizer.model_max_length = 128
    tokenizer.save_pretrained(str(model_dir))

    summary = {
        "base_model": model_name,
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "label_names": label_names,
        "max_length": 128,
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "seed": 42,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "validation_results": results,
    }

    (reports_dir / "distilbert_training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (reports_dir / "distilbert_training_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2),
        encoding="utf-8",
    )

    print("\nBEST MODEL VALIDATION RESULTS")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Macro F1: {results['eval_macro_f1']:.4f}")
    print("\nModel saved to:", model_dir)
    print("Full training completed successfully.")


if __name__ == "__main__":
    main()