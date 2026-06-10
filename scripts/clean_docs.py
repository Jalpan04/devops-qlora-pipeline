import os
import sys

# Ensure dependencies can be imported
try:
    from bs4 import BeautifulSoup
    import markdownify
except ImportError:
    print("Required libraries 'beautifulsoup4' and/or 'markdownify' not found.")
    print("Please run: pip install beautifulsoup4 markdownify")
    sys.exit(1)

def clean_html_to_markdown(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Remove noise elements
    noise_tags = ["nav", "footer", "header", "script", "style", "aside", "iframe"]
    for tag in noise_tags:
        for element in soup.find_all(tag):
            element.decompose()
            
    # Remove standard sidebar or navigation classes
    # e.g., elements with class containing 'nav', 'footer', 'sidebar', 'menu', 'header'
    for element in soup.find_all(class_=lambda c: c and any(x in c.lower() for x in ['sidebar', 'footer', 'header', 'navigation', 'menu', 'banner'])):
        # But make sure we don't accidentally delete main content if it has a class like 'content-header'
        if element.name not in ['main', 'article', 'div']:
            element.decompose()
        elif element.name == 'div' and any(x in element.get('id', '').lower() for x in ['sidebar', 'navigation']):
            element.decompose()

    # 2. Extract main/article content if present, to reduce noise further
    main_content = None
    for target in ["main", "article", "[role=main]", "#main-content"]:
        if target.startswith("["):
            attr, val = target.strip("[]").split("=")
            main_content = soup.find(attrs={attr: val})
        elif target.startswith("#"):
            main_content = soup.find(id=target[1:])
        else:
            main_content = soup.find(target)
        if main_content:
            break
            
    content_soup = main_content if main_content else soup
    
    # 3. Convert HTML to Markdown using markdownify
    # Custom options to keep formatting clean
    md_content = markdownify.markdownify(
        str(content_soup),
        heading_style="ATX", # Use # instead of underline
        code_language_callback=lambda el: el.get('class', [''])[0].replace('language-', '') if el.get('class') else ''
    )
    
    # Clean up empty lines and spaces
    md_lines = [line.rstrip() for line in md_content.split('\n')]
    cleaned_md = '\n'.join(md_lines)
    import re
    cleaned_md = re.sub(r'\n{3,}', '\n\n', cleaned_md)
    
    return cleaned_md.strip()

def main():
    input_dir = "01_raw_data/official_docs"
    output_dir = "02_clean_data/official_docs_md"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist. Please run download_docs.py first.")
        return
        
    for filename in os.listdir(input_dir):
        if not filename.endswith(".html"):
            continue
            
        in_path = os.path.join(input_dir, filename)
        out_name = filename.replace(".html", ".md")
        out_path = os.path.join(output_dir, out_name)
        
        print(f"Converting and cleaning: {filename}...")
        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
            
        cleaned_md = clean_html_to_markdown(html)
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Official Documentation: {filename.replace('.html', '').replace('_', ' ').title()}\n\n")
            f.write(cleaned_md)
            f.write("\n")
            
        print(f"Saved cleaned markdown to {out_path}")

if __name__ == "__main__":
    main()
