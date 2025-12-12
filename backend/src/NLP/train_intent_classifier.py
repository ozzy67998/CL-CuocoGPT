# train_intent_classifier.py
from datasets import load_dataset, ClassLabel, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate, numpy as np, json, os

MODEL_NAME = "bert-base-multilingual-cased"  # or "neuralmind/bert-base-portuguese-cased"
DATA_PATH = "../../data/raw/bert_intent_training.csv"  # text,label
OUT_DIR = "../../models/intent-bert"

# 1) Load and split dataset
ds = load_dataset("csv", data_files=DATA_PATH)["train"].train_test_split(test_size=0.2, seed=42)
# Collect unique labels from train and test splits reliably (avoid Column addition)
train_labels = ds["train"].unique("label")
test_labels = ds["test"].unique("label")
labels = sorted(set(train_labels) | set(test_labels))
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

def encode_label(example):
    example["labels"] = label2id[example["label"]]
    return example

ds = ds.map(encode_label)

# 2) Tokenizer and model
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(ex):
    return tok(ex["text"], truncation=True)
tokenized = ds.map(tokenize, batched=True, remove_columns=["text", "label"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# 3) Training setup
data_collator = DataCollatorWithPadding(tokenizer=tok)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, y = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": metric.compute(predictions=preds, references=y)["accuracy"]}

# Create TrainingArguments with backward compatibility for older transformers versions
try:
    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=42,
        logging_steps=50,
    )
except TypeError:
    # Fallback for older transformers that don't support evaluation/save strategies
    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        seed=42,
        logging_steps=50,
        save_steps=500,
    )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# Evaluate at the end to report accuracy even if in-training eval wasn't configured
eval_metrics = trainer.evaluate()
print({"final_eval": eval_metrics})

# 4) Save model + label map
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)
with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)