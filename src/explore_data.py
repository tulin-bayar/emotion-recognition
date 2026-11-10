from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load the dataset with predefined train, validation and test sets.
# (Hazır eğitim, doğrulama ve test bölümleriyle veri setini yükle.)
dataset = load_dataset("dair-ai/emotion", "split")

print("DATASET STRUCTURE")
print(dataset)

# Read emotion names from the dataset.
# (Duygu isimlerini veri setinden oku.)
label_names = dataset["train"].features["label"].names

print("\nEMOTION LABELS")
for number, name in enumerate(label_names):
    print(f"{number}: {name}")

# Convert the training set into a pandas DataFrame.
# (Eğitim verisini pandas tablosuna dönüştür.)
train_df = dataset["train"].to_pandas()

# Add readable emotion names next to numeric labels.
# (Sayısal etiketlerin yanına okunabilir duygu isimlerini ekle.)
train_df["emotion"] = train_df["label"].apply(
    lambda label: label_names[label]
)

print("\nFIRST 5 EXAMPLES")
print(train_df.head().to_string(index=False))

print("\nEXAMPLES PER EMOTION")
print(train_df["emotion"].value_counts())

print("\nMISSING VALUES")
print(train_df.isnull().sum())

# Create the reports folder relative to the project location.
# (Projenin konumuna göre reports klasörünü oluştur.)
project_root = Path(__file__).resolve().parent.parent
reports_dir = project_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# Count examples in the dataset's label order.
# (Örnekleri veri setinin etiket sırasına göre say.)
emotion_counts = train_df["emotion"].value_counts().reindex(label_names)

# Draw the emotion distribution.
# (Duygu dağılımını çiz.)
fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(emotion_counts.index, emotion_counts.values, color="steelblue")
ax.bar_label(bars, padding=3)

ax.set_title("Emotion Distribution in the Training Set")
ax.set_xlabel("Emotion")
ax.set_ylabel("Number of Examples")
ax.set_ylim(0, emotion_counts.max() * 1.15)

fig.tight_layout()

# Save the chart and the counts.
# (Grafiği ve sayıları kaydet.)
fig.savefig(reports_dir / "emotion_distribution.png", dpi=150)
plt.close(fig)

emotion_counts.to_csv(
    reports_dir / "emotion_counts.csv",
    header=["count"],
    index_label="emotion"
)

print("\nChart and counts saved to:", reports_dir)

# Check for empty or whitespace-only messages.
# (Boş veya yalnızca boşluk içeren mesajları kontrol et.)
empty_texts = train_df["text"].fillna("").str.strip().eq("").sum()

# Count repeated texts after their first occurrence.
# (İlk görülmeden sonraki tekrar eden metinleri say.)
duplicate_texts = train_df.duplicated(subset=["text"]).sum()

# Count repeated text-label pairs.
# (Hem metni hem etiketi aynı olan tekrarları say.)
duplicate_rows = train_df.duplicated(subset=["text", "label"]).sum()

# Find texts assigned to more than one emotion.
# (Birden fazla duygu etiketi verilmiş metinleri bul.)
labels_per_text = train_df.groupby("text")["label"].nunique()
conflicting_texts = (labels_per_text > 1).sum()

print("\nTRAINING DATA QUALITY")
print(f"Empty texts: {empty_texts}")
print(f"Duplicate texts: {duplicate_texts}")
print(f"Duplicate text-label pairs: {duplicate_rows}")
print(f"Texts with conflicting labels: {conflicting_texts}")

# Get texts that have multiple emotion labels.
# (Birden fazla duygu etiketi olan metinleri al.)
conflicting_messages = labels_per_text[labels_per_text > 1].index

# Collect all rows containing these texts.
# (Bu metinleri içeren tüm satırları seç.)
conflicting_df = train_df[
    train_df["text"].isin(conflicting_messages)
].sort_values("text")

# Save the conflicting examples for review.
# (Çelişkili örnekleri incelemek için kaydet.)
conflicting_df.to_csv(
    reports_dir / "conflicting_labels.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCONFLICTING LABEL EXAMPLES")
print(conflicting_df.head(6).to_string(index=False))

# Compare exact texts across dataset splits.
# (Veri bölümleri arasındaki birebir aynı metinleri karşılaştır.)
train_texts = set(dataset["train"]["text"])
validation_texts = set(dataset["validation"]["text"])
test_texts = set(dataset["test"]["text"])

print("\nSPLIT OVERLAP CHECK")
print(f"Train / validation: {len(train_texts & validation_texts)}")
print(f"Train / test: {len(train_texts & test_texts)}")
print(f"Validation / test: {len(validation_texts & test_texts)}")