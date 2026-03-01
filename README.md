# Multilingual Translation SLM

This project implements a Small Language Model (SLM) for translating English to 5 Indian languages:
- Hindi
- Kannada
- Tamil
- Telugu
- Marathi

## Features
- **Data Loading**: Uses the `ai4bharat/samanantar` dataset (or subsets) via Hugging Face.
- **Model**: Fine-tuning `google/mt5-small` (a multilingual T5 model) for translation.
- **Interface**: A Streamlit web app for easy interaction.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training
Run the training script to fine-tune the model:
```bash
python train.py
```

### Interface
Run the Streamlit app:
```bash
streamlit run app.py
```

## Where to see the data and model
- **Training data (loaded dataset)**: After you run `python train.py`, the data used for training is saved in the **`training_data`** folder in this project:
  - **`training_data/training_data.csv`** — CSV with columns `english`, `target`, and `lang` (one row per sentence pair).
- **Trained model**: After training finishes, the fine-tuned model is saved in the **`mt5-translation-model`** folder (config, tokenizer, and weights). The Streamlit app uses this when you select "Fine-tuned Model (Local)".
- **Source dataset**: The raw data is fetched from Hugging Face (**ai4bharat/samanantar**); it is not stored in a separate folder until you run training, which writes the sampled data to `training_data/`.
