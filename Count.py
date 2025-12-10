import os
import glob
from tqdm import tqdm
from transformers import AutoTokenizer

# --- CONFIGURATION ---
# Base path to your data folder
# Based on your image: Final_v1/AIFORTHAI-LST20Corpus/LST20_Corpus_files/data/
# CHECK THIS: I assumed the cut-off folder name is "LST20_Corpus_files"
BASE_DATA_PATH = os.path.join("Final_v1", "AIFORTHAI-LST20Corpus", "LST20_Corpus_final", "data")

# The specific subfolders you want to scan
TARGET_FOLDERS = ["new_fix", "old_fix"]

# The model tokenizer
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"

def load_text_from_folder(base_path, folders):
    """
    Recursively finds all .txt files in the specified subfolders and reads their content.
    """
    all_text_lines = []
    
    for folder in folders:
        # Construct the full path to the subfolder
        folder_path = os.path.join(base_path, folder)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue

        # Recursive glob to find all .txt files inside
        search_path = os.path.join(folder_path, "**", "*.txt")
        files = glob.glob(search_path, recursive=True)
        
        print(f"Found {len(files)} text files in '{folder}'...")
        
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Read lines, strip whitespace, and ignore empty lines
                    lines = [line.strip() for line in f if line.strip()]
                    all_text_lines.extend(lines)
            except Exception as e:
                print(f"Skipping file {file_path}: {e}")
                
    return all_text_lines

def main():
    print(f"--- Token Counting for folders: {TARGET_FOLDERS} ---")
    
    if not os.path.exists(BASE_DATA_PATH):
        print(f"Error: Base path not found: {BASE_DATA_PATH}")
        print("Please check the folder name inside the script (I assumed 'LST20_Corpus_files').")
        return

    # 1. Load the raw text lines
    print("Reading files (this might take a moment)...")
    sentences = load_text_from_folder(BASE_DATA_PATH, TARGET_FOLDERS)
    
    if not sentences:
        print("No text data found. Check your paths.")
        return
        
    print(f"Successfully loaded {len(sentences):,} lines of text.")

    # 2. Initialize Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    total_subwords = 0
    
    # 3. Count
    print("Tokenizing and counting...")
    for sentence in tqdm(sentences):
        # Tokenize the line. WangchanBERTa handles raw text directly.
        # add_special_tokens=False ensures we count just the content tokens, 
        # not the [CLS] and [SEP] added for model input.
        encoding = tokenizer(sentence, add_special_tokens=False)
        total_subwords += len(encoding["input_ids"])

    print("\n" + "="*40)
    print("DATASET STATISTICS")
    print("="*40)
    print(f"Total Text Lines/Segments: {len(sentences):,}")
    print(f"Total Subword Tokens:      {total_subwords:,}")
    print("="*40)

if __name__ == "__main__":
    main()