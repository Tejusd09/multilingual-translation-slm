from datasets import load_dataset
import pandas as pd
import os

def load_translation_data(lang_pair, split='train', max_samples=1000):
    """
    Load translation data for a given language pair.
    
    Args:
        lang_pair (str): Language pair code (e.g., 'hi', 'kn', 'ta', 'te', 'mr'). 
                         Note: Samanantar usually uses just the target lang code if source is English.
        split (str): Dataset split to load.
        max_samples (int): Number of samples to load (streaming mode).
        
    Returns:
        pd.DataFrame: DataFrame with 'english' and 'target' columns.
    """
    print(f"Loading {max_samples} samples for English to {lang_pair}...")
    
    # Using ai4bharat/samanantar dataset
    # Configuration usually is the language code for the target language (e.g. 'hi')
    try:
        # Stream the dataset to avoid downloading everything
        dataset = load_dataset("ai4bharat/samanantar", lang_pair, split=split, streaming=True)
        
        data = []
        count = 0
        for example in dataset:
            # Structure usually has 'idx' and translation pairs
            # But specific structure might vary. Let's inspect usually it is 'source' and 'target' or similar
            # Samanantar HF structure: features usually are 'idx', 'src', 'tgt'
            # Let's verify by try-except or just standardizing
            
            src_text = example.get('src') or example.get('english')
            tgt_text = example.get('tgt') or example.get(lang_pair)
            
            if src_text and tgt_text:
                data.append({
                    'english': src_text,
                    'target': tgt_text,
                    'lang': lang_pair
                })
                count += 1
                if count >= max_samples:
                    break
                    
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} samples.")
        return df
        
    except Exception as e:
        print(f"Error loading dataset for {lang_pair}: {e}")
        return pd.DataFrame()

def save_data_locally(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    # Test loading for Hindi
    df_hi = load_translation_data("hi", max_samples=100)
    if not df_hi.empty:
        print(df_hi.head())
