import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from data_loader import load_translation_data, save_data_locally
import os

# Configuration (light on laptop: small batches, fewer samples, shorter sequences)
MODEL_CHECKPOINT = "google/mt5-small"
MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 2e-5
OUTPUT_DIR = "mt5-translation-model"
DATA_DIR = "training_data"
SAMPLES_PER_LANG = 50  # Fewer samples = less data loading and memory

# Target languages mapping
LANG_CODES = {
    'Hindi': 'hi',
    'Kannada': 'kn',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Marathi': 'mr'
}

def preprocess_function(examples, tokenizer):
    inputs = [f"translate English to {ex['lang']}: " + ex['english'] for ex in examples]
    targets = [ex['target'] for ex in examples]
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model():
    print("Loading data...")
    data_path = os.path.join(DATA_DIR, "training_data.csv")
    full_df = None

    # Prefer local CSV (survives restarts, no re-download)
    if os.path.isfile(data_path):
        print(f"Loading from local data: {data_path}")
        full_df = pd.read_csv(data_path)
        if "lang" not in full_df.columns:
            full_df["lang"] = "hi"  # default if missing
        full_df = full_df[["english", "target", "lang"]].dropna()
        print(f"Loaded {len(full_df)} samples from CSV.")
    if full_df is None or full_df.empty:
        all_data = []
        for lang_name, lang_code in LANG_CODES.items():
            df = load_translation_data(lang_code, max_samples=SAMPLES_PER_LANG)
            if not df.empty:
                all_data.append(df)
        if not all_data:
            print("No data loaded. Check internet connection or dataset availability.")
            return
        full_df = pd.concat(all_data)
        os.makedirs(DATA_DIR, exist_ok=True)
        save_data_locally(full_df, data_path)
        print(f"Training data saved to: {os.path.abspath(data_path)}")

    dataset = Dataset.from_pandas(full_df)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Resume from latest checkpoint if present (e.g. after PC restart)
    resume_from_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            resume_from_checkpoint = os.path.join(OUTPUT_DIR, latest)
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    print(f"Initializing model: {MODEL_CHECKPOINT}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    load_from = resume_from_checkpoint if resume_from_checkpoint else MODEL_CHECKPOINT
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = AutoModelForSeq2SeqLM.from_pretrained(load_from)
    
    def preprocess_wrapper(examples):
        # We need to restructure the batch for preprocessing
        # The dataset 'examples' is a dictionary of lists
        ex_list = []
        for i in range(len(examples['english'])):
            ex_list.append({
                'english': examples['english'][i],
                'target': examples['target'][i],
                'lang': examples['lang'][i]
            })
        return preprocess_function(ex_list, tokenizer)

    print("Tokenizing data...")
    tokenized_datasets = dataset.map(preprocess_wrapper, batched=True)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,  # effective batch 8 with less memory per step
        weight_decay=0.01,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        dataloader_num_workers=0,  # avoid extra processes on laptop
        report_to="none",  # disable wandb so training runs without login
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()
