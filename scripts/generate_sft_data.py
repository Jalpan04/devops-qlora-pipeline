import os
import json
import time
import sys
import logging
import concurrent.futures
from typing import List, Dict, Optional
import urllib.request
from threading import Lock

# Configuration
PROVIDER = "gemini"  # "gemini" or "ollama"
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
GEMINI_MODEL = "gemini-2.0-flash"

MAX_WORKERS = 1 if PROVIDER == "gemini" else 4  # Number of parallel requests (Gemini free tier has strict RPM/TPM limits)
MAX_CHARS_PER_CHUNK = 15000  # Size of text blocks to send to LLM
PROGRESS_FILE = ".processing_progress.json"
OUTPUT_FILE = "03_ready_for_qlora/instructions.jsonl"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("generator.log"), logging.StreamHandler()]
)

# SDK Imports
USE_NEW_SDK = False
if PROVIDER == "gemini":
    try:
        from google import genai
        from google.genai import types
        USE_NEW_SDK = True
    except ImportError:
        try:
            import google.generativeai as genai_legacy
            USE_NEW_SDK = False
        except ImportError:
            logging.error("Required Google GenAI library not found. Run: pip install google-genai")
            sys.exit(1)

class ProgressTracker:
    def __init__(self, filename):
        self.filename = filename
        self.processed_files = self._load()
        self.lock = Lock()

    def _load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return set(json.load(f))
        return set()

    def is_processed(self, filepath):
        return filepath in self.processed_files

    def mark_done(self, filepath):
        with self.lock:
            self.processed_files.add(filepath)
            with open(self.filename, 'w') as f:
                json.dump(list(self.processed_files), f)

def get_client():
    if PROVIDER == "ollama":
        return "ollama"
            
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
        
    if USE_NEW_SDK:
        return genai.Client(api_key=api_key)
    else:
        genai_legacy.configure(api_key=api_key)
        return genai_legacy

def call_llm(client, content: str, source_name: str) -> Optional[List[Dict]]:
    system_instruction = "You are an expert DevOps training data generator. Generate output in valid raw JSON array format. No markdown blocks, no emojis."
    
    prompt = f"""Content: {content}
Source: {source_name}

Generate exactly 5 high-quality, complex DevOps instruction-response pairs based on the content above. 
You MUST provide one pair for each style:
1. Direct Command (e.g. "Create a script to...")
2. Troubleshooting (Fixing an error)
3. Conceptual (Comparing features)
4. Refactoring (Optimizing code)
5. Standard How-To (Typical user query)

Return ONLY a JSON array:
[ {{"user": "...", "assistant": "..."}}, ... ]
"""

    max_retries = 4
    backoff = 3.0  # Initial sleep time in seconds
    
    for attempt in range(max_retries):
        try:
            if PROVIDER == "ollama":
                payload = {
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False, "format": "json", "options": {"temperature": 0.3}
                }
                req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as res:
                    response = json.loads(res.read().decode("utf-8"))
                    text = response["message"]["content"]
            else:
                if USE_NEW_SDK:
                    res = client.models.generate_content(
                        model=GEMINI_MODEL, contents=prompt,
                        config=types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json")
                    )
                    text = res.text
                else:
                    model = client.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system_instruction)
                    res = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                    text = res.text

            data = json.loads(text)
            # Handle cases where LLM wraps list in a key
            if isinstance(data, dict):
                for k in ["pairs", "qa", "instructions"]:
                    if k in data: return data[k]
                return [data] if "user" in data else None
            return data
            
        except Exception as e:
            err_msg = str(e)
            logging.warning(f"LLM call failed for {source_name} (attempt {attempt + 1}/{max_retries}): {err_msg}")
            if attempt < max_retries - 1:
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_time = 35.0
                    logging.info(f"Rate limit hit (429). Sleeping for {sleep_time} seconds to reset the window...")
                else:
                    import random
                    sleep_time = (backoff * (2 ** attempt)) + random.uniform(0.5, 2.0)
                    logging.info(f"Retrying {source_name} in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                # Last resort fallback if we have a parsed JSON structure but formatting/decoding issue
                if 'text' in locals():
                    try:
                        clean_text = text.strip()
                        if clean_text.startswith("```json"):
                            clean_text = clean_text[7:]
                        if clean_text.endswith("```"):
                            clean_text = clean_text[:-3]
                        data = json.loads(clean_text.strip())
                        if isinstance(data, dict):
                            for k in ["pairs", "qa", "instructions"]:
                                if k in data: return data[k]
                            return [data] if "user" in data else None
                        return data
                    except:
                        pass
                return None

def process_single_file(filepath: str, filename: str, client, tracker: ProgressTracker, writer_lock: Lock):
    if tracker.is_processed(filepath):
        return 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        full_content = f.read()

    # Skip extremely short or empty files
    if len(full_content.strip()) < 50:
        logging.info(f"Skipping {filename}: Content too short.")
        tracker.mark_done(filepath)
        return 0

    # Split content into chunks if too large to ensure we cover the whole file
    chunks = [full_content[i:i+MAX_CHARS_PER_CHUNK] for i in range(0, len(full_content), MAX_CHARS_PER_CHUNK)]
    
    local_qa_count = 0
    success = False
    
    for i, chunk in enumerate(chunks[:2]): # Limit to first 2 chunks to prevent infinite costs
        qa_pairs = call_llm(client, chunk, f"{filename}_part_{i}")
        
        if qa_pairs is not None:
            # The API call succeeded (even if it returned empty list/invalid format, it didn't throw connection/503/429 error)
            success = True
            
        if qa_pairs and isinstance(qa_pairs, list):
            with writer_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                    for pair in qa_pairs:
                        if "user" in pair and "assistant" in pair:
                            record = {
                                "messages": [
                                    {"role": "system", "content": "You are an expert DevOps AI."},
                                    {"role": "user", "content": pair["user"]},
                                    {"role": "assistant", "content": pair["assistant"]}
                                ]
                            }
                            out_f.write(json.dumps(record) + "\n")
                            local_qa_count += 1
            
    if success:
        tracker.mark_done(filepath)
    else:
        logging.warning(f"File {filename} failed all LLM calls due to API/network errors. Will retry next run.")
        
    return local_qa_count


def main():
    client = get_client()
    tracker = ProgressTracker(PROGRESS_FILE)
    writer_lock = Lock()
    
    clean_dirs = ["02_clean_data/github_md", "02_clean_data/manpages_clean", "02_clean_data/official_docs_md"]
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    files_to_process = []
    for directory in clean_dirs:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                path = os.path.join(directory, f)
                if os.path.isfile(path) and not tracker.is_processed(path):
                    files_to_process.append((path, f))

    if not files_to_process:
        logging.info("No new files to process.")
        return

    logging.info(f"Starting processing for {len(files_to_process)} new files using {MAX_WORKERS} workers...")

    total_generated = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, fp, fn, client, tracker, writer_lock): fn for fp, fn in files_to_process}
        
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                count = future.result()
                total_generated += count
                if count > 0:
                    logging.info(f"Completed {filename}: Generated {count} pairs.")
            except Exception as e:
                logging.error(f"File {filename} generated an exception: {e}")

    logging.info(f"Generation complete. Total pairs: {total_generated}")

if __name__ == "__main__":
    main()