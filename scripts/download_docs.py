import urllib.request
import os
import time

def download_url(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "DevOps-Dataset-Builder")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read()
            with open(output_path, "wb") as f:
                f.write(html)
            print(f"Successfully saved to {output_path}")
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    docs_to_download = {
        "python_os.html": "https://docs.python.org/3/library/os.html",
        "python_subprocess.html": "https://docs.python.org/3/library/subprocess.html",
        "kubernetes_kubectl_cheatsheet.html": "https://kubernetes.io/docs/reference/kubectl/cheatsheet/",
        "docker_cli_reference.html": "https://docs.docker.com/reference/cli/docker/"
    }
    
    output_dir = "01_raw_data/official_docs"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, url in docs_to_download.items():
        out_path = os.path.join(output_dir, filename)
        if os.path.exists(out_path):
            print(f"{filename} already exists, skipping.")
            continue
        download_url(url, out_path)
        time.sleep(2) # Be gentle
        
    print("Completed official docs download phase.")

if __name__ == "__main__":
    main()
