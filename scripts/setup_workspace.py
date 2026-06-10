import os

def main():
    dirs = [
        "01_raw_data/github_repos",
        "01_raw_data/linux_manpages",
        "01_raw_data/official_docs",
        "01_raw_data/hf_datasets",
        "02_clean_data/github_md",
        "02_clean_data/manpages_clean",
        "02_clean_data/official_docs_md",
        "02_clean_data/code_scripts",
        "03_ready_for_qlora",
        "scripts"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

if __name__ == "__main__":
    main()
