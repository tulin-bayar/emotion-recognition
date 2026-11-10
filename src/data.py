from datasets import load_dataset


def load_emotion_data():
    # Load the original dataset.
    # (Orijinal veri setini yükle.)
    dataset = load_dataset("dair-ai/emotion", "split")

    label_names = dataset["train"].features["label"].names

    train_df = dataset["train"].to_pandas()
    validation_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    original_train_size = len(train_df)
    original_validation_size = len(validation_df)

    # Keep the test set unchanged.
    # (Test bölümünü değiştirmeden koru.)
    test_texts = set(test_df["text"])

    validation_df = validation_df[
        ~validation_df["text"].isin(test_texts)
    ].copy()

    # Remove training texts found in either evaluation set.
    # (Değerlendirme bölümlerindeki metinleri eğitimden çıkar.)
    evaluation_texts = test_texts | set(validation_df["text"])

    train_df = train_df[
        ~train_df["text"].isin(evaluation_texts)
    ].copy()

    print("Removed training rows:", original_train_size - len(train_df))
    print(
        "Removed validation rows:",
        original_validation_size - len(validation_df)
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        label_names,
    )


if __name__ == "__main__":
    train_df, validation_df, test_df, label_names = load_emotion_data()

    print("\nFINAL SPLIT SIZES")
    print("Train:", len(train_df))
    print("Validation:", len(validation_df))
    print("Test:", len(test_df))

    train_texts = set(train_df["text"])
    validation_texts = set(validation_df["text"])
    test_texts = set(test_df["text"])

    assert not (train_texts & validation_texts)
    assert not (train_texts & test_texts)
    assert not (validation_texts & test_texts)

    print("\nChecks passed: no exact text overlap between splits.")