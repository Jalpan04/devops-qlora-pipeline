import json
import os
import hashlib

def get_content_hash(text):
    # Normalize whitespace to catch minor formatting differences
    normalized = "".join(text.split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def main():
    input_file = "01_raw_data/hf_datasets/raw_scripts.jsonl"
    output_file = "02_clean_data/code_scripts/clean_scripts.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist. Please run download_hf.py first.")
        return
        
    seen_hashes = set()
    total_records = 0
    saved_records = 0
    short_skipped = 0
    long_skipped = 0
    dup_skipped = 0
    
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            total_records += 1
            data = json.loads(line)
            code = data["code"]
            
            # Filter by line count: between 3 and 2000 lines (3 to preserve shorter YAMLs/Dockerfiles)
            line_count = len(code.splitlines())
            if line_count < 3:
                short_skipped += 1
                continue
            if line_count > 2000:
                long_skipped += 1
                continue
                
            # Deduplication using content hash
            content_hash = get_content_hash(code)
            if content_hash in seen_hashes:
                dup_skipped += 1
                continue
            seen_hashes.add(content_hash)
            
            # Save clean record
            f_out.write(json.dumps(data) + "\n")
            saved_records += 1
            
    print(f"Cleaning complete.")
    print(f"Total processed: {total_records}")
    print(f"Saved: {saved_records}")
    print(f"Skipped (too short): {short_skipped}")
    print(f"Skipped (too long): {long_skipped}")
    print(f"Skipped (duplicates): {dup_skipped}")

if __name__ == "__main__":
    main()
