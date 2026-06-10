import os
import re

def clean_manpage_content(content):
    # 1. Remove backspace formatting backspaces (e.g. o\x08ob\x08bl\x08ld\x08d -> bold)
    content = re.sub(r'.\x08', '', content)
    
    # 2. Split into lines
    lines = content.split('\n')
    
    # Standard sections we want to KEEP
    keep_headers = {
        "NAME", "SYNOPSIS", "DESCRIPTION", "OPTIONS", "EXAMPLES", 
        "EXIT STATUS", "ENVIRONMENT", "FILES", "NOTES", "DIAGNOSTICS", "COMMANDS"
    }
    
    # Sections we explicitly want to DISCARD
    discard_headers = {
        "AUTHOR", "AUTHORS", "COPYRIGHT", "REPORTING BUGS", "SEE ALSO", 
        "COLOPHON", "HISTORY", "LICENSE", "AVAILABILITY"
    }
    
    cleaned_lines = []
    current_section = None
    skip_current_section = False
    
    # Helper to check if a line is a section header
    # Usually section headers are uppercase, start at column 0, and are not empty
    def is_header(line):
        if not line:
            return False
        # Section headers are usually fully capitalized words with spaces
        # Let's match line that is uppercase and doesn't start with space
        # e.g., "NAME" or "SEE ALSO" or "REPORTING BUGS"
        if re.match(r'^[A-Z][A-Z\s\-]+$', line):
            return True
        return False

    for line in lines:
        stripped = line.strip()
        if is_header(stripped):
            current_section = stripped
            if current_section in discard_headers:
                skip_current_section = True
            elif current_section in keep_headers:
                skip_current_section = False
            else:
                # If it's a section we don't recognize, default to keeping it (or skipping if it's after AUTHOR)
                if current_section.startswith("AUTHOR") or "BUG" in current_section or "COPYRIGHT" in current_section:
                    skip_current_section = True
                else:
                    skip_current_section = False
                    
            if not skip_current_section:
                cleaned_lines.append("")
                cleaned_lines.append(line)
            continue
            
        if not skip_current_section:
            cleaned_lines.append(line)
            
    # Combine lines and clean up whitespace
    cleaned_text = '\n'.join(cleaned_lines)
    # Remove multiple consecutive blank lines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def main():
    input_dir = "01_raw_data/linux_manpages"
    output_dir = "02_clean_data/manpages_clean"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist. Please run download_manpages.py first.")
        return
        
    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue
            
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        
        print(f"Cleaning manpage: {filename}...")
        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        cleaned = clean_manpage_content(content)
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
            f.write("\n")
            
        print(f"Saved cleaned manpage to {out_path}")

if __name__ == "__main__":
    main()
