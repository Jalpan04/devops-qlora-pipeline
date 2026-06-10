import json
import re
import os

def clean_readme(content):
    # 1. Remove base64 encoded images/assets
    # e.g., ![img](data:image/png;base64,iVBOR...)
    content = re.sub(r'!\[[^\]]*\]\(data:[^;]+;base64,[A-Za-z0-9+/=\s]+\)', '', content)
    
    # 2. Remove standard badge patterns: [! [alt] (badge-url)] (link-url) or ! [alt] (badge-url)
    # We can detect SVG badges, shield.io, travis-ci, codecov, badge.fury, etc.
    badge_patterns = [
        r'\[!\[[^\]]*\]\([^)]*badge[^)]*\)\]\([^)]*\)',
        r'\[!\[[^\]]*\]\([^)]*shields\.io[^)]*\)\]\([^)]*\)',
        r'!\[[^\]]*\]\([^)]*badge[^)]*\)',
        r'!\[[^\]]*\]\([^)]*shields\.io[^)]*\)',
        r'!\[[^\]]*\]\([^)]*travis-ci[^)]*\)',
        r'!\[[^\]]*\]\([^)]*github/workflow/[^)]*\)'
    ]
    for pattern in badge_patterns:
        content = re.compile(pattern, re.IGNORECASE).sub('', content)
        
    # 3. Strip HTML badges/links that contain image tags with shields/badges
    content = re.sub(r'<a href="[^"]*"><img src="[^"]*(badge|shield)[^"]*"[^>]*></a>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<img src="[^"]*(badge|shield)[^"]*"[^>]*>', '', content, flags=re.IGNORECASE)
    
    # 4. Remove contributor tables or list of contributors.
    # Contributor tables often look like: | <a href=...><img src=... width=...><br /><sub><b>Name</b></sub></a> |
    # We can look for table blocks containing lots of avatar images.
    lines = content.split('\n')
    cleaned_lines = []
    in_contributors_section = False
    skip_table = False
    
    for line in lines:
        # Detect if we enter a contributors section
        if re.search(r'##\s+Contributors', line, re.IGNORECASE) or re.search(r'##\s+Credits', line, re.IGNORECASE):
            in_contributors_section = True
            cleaned_lines.append(line)
            continue
            
        if in_contributors_section:
            # If we see another header, we exit the contributors section
            if line.startswith('##'):
                in_contributors_section = False
                skip_table = False
            # Detect tables or lists with many avatars/github profiles
            elif 'githubusercontent.com/u/' in line or 'avatars' in line or 'contrib' in line:
                # Skip lines that are clearly contributor listings
                continue
                
        # Also, generic contributor table detection (lots of td/img/href)
        if line.strip().startswith('|') and ('<img' in line or 'avatars' in line) and ('width=' in line or 'height=' in line):
            continue
            
        cleaned_lines.append(line)
        
    cleaned_content = '\n'.join(cleaned_lines)
    
    # 5. Clean up multiple empty lines
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    return cleaned_content.strip()

def main():
    input_file = "01_raw_data/github_repos/readmes.jsonl"
    output_dir = "02_clean_data/github_md"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist. Please run download_github.py first.")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            repo_name = data["repo_name"]
            readme_content = data["readme_content"]
            
            print(f"Cleaning README for {repo_name}...")
            cleaned = clean_readme(readme_content)
            
            # Save as separate Markdown file in 02_clean_data/github_md
            # Replace / with _ for filename
            safe_name = repo_name.replace("/", "_") + ".md"
            out_path = os.path.join(output_dir, safe_name)
            
            with open(out_path, "w", encoding="utf-8") as out_f:
                # Add a metadata header for the LLM context
                out_f.write(f"# Repository: {repo_name}\n")
                out_f.write(f"Stars: {data['stars']}\n")
                if data.get('description'):
                    out_f.write(f"Description: {data['description']}\n")
                out_f.write("\n---\n\n")
                out_f.write(cleaned)
                out_f.write("\n")
                
            print(f"Saved cleaned markdown to {out_path}")

if __name__ == "__main__":
    main()
