import os
import json

def main():
    output_file = "03_ready_for_qlora/pretraining.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    clean_dirs = [
        "02_clean_data/github_md",
        "02_clean_data/manpages_clean",
        "02_clean_data/official_docs_md"
    ]
    
    code_scripts_file = "02_clean_data/code_scripts/clean_scripts.jsonl"
    
    total_docs = 0
    total_bytes = 0
    
    print("Preparing Continued Pre-Training (CPT) dataset...")
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        # 1. Process markdown/text files
        for directory in clean_dirs:
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist. Skipping.")
                continue
                
            print(f"Concatenating files from {directory}...")
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as in_f:
                        content = in_f.read()
                        
                    # Write document header and content
                    header = f"\n\n--- DOCUMENT START: {filename} ---\n"
                    out_f.write(header)
                    out_f.write(content)
                    out_f.write("\n--- DOCUMENT END ---\n")
                    
                    total_docs += 1
                    total_bytes += len(content.encode('utf-8'))
                    
        # 2. Process code scripts
        if os.path.exists(code_scripts_file):
            print(f"Concatenating scripts from {code_scripts_file}...")
            with open(code_scripts_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    code = data["code"]
                    repo = data.get("repo_name", "unknown")
                    path = data.get("path", "unknown")
                    lang = data.get("language", "unknown")
                    
                    header = f"\n\n--- SCRIPT START: {path} (Repo: {repo}, Lang: {lang}) ---\n"
                    out_f.write(header)
                    out_f.write(code)
                    out_f.write("\n--- SCRIPT END ---\n")
                    
                    total_docs += 1
                    total_bytes += len(code.encode('utf-8'))
        else:
            print(f"Warning: Code scripts file {code_scripts_file} not found.")
            
    print(f"\nCPT dataset preparation complete.")
    print(f"Total documents/scripts compiled: {total_docs}")
    print(f"Total dataset size: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"CPT file saved to: {output_file}")

if __name__ == "__main__":
    main()
