import os
import urllib.request
import zipfile
import json
import io

def download_and_extract_zip(url, repo_name, allowed_extensions, output_jsonl_file, max_files=5000):
    max_retries = 3
    zip_data = None
    for attempt in range(1, max_retries + 1):
        print(f"Downloading source zip for {repo_name} from {url} (Attempt {attempt}/{max_retries})...")
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "DevOps-Dataset-Builder")
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                zip_data = response.read()
            print(f"Downloaded zip successfully. Extracting scripts...")
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"Error downloading {url}: HTTP Error 404: Not Found")
                return 0
            print(f"Attempt {attempt}/{max_retries} failed for {repo_name}: {e}")
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed for {repo_name}: {e}")
            
        if attempt < max_retries:
            time.sleep(3)
    else:
        print(f"Failed to download zip for {repo_name} after {max_retries} attempts.")
        return 0
        
    extracted_count = 0
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        for file_info in z.infolist():
            if file_info.is_dir():
                continue
            
            # Check if file has an allowed extension or matches special naming
            filename_lower = file_info.filename.lower()
            _, ext = os.path.splitext(file_info.filename)
            ext = ext.lower().lstrip('.')
            
            is_dockerfile = filename_lower.endswith('/dockerfile') or filename_lower == 'dockerfile'
            is_jenkinsfile = filename_lower.endswith('/jenkinsfile') or filename_lower == 'jenkinsfile'
            
            if ext in allowed_extensions or (is_dockerfile and "dockerfile" in allowed_extensions) or (is_jenkinsfile and "jenkinsfile" in allowed_extensions):
                try:
                    with z.open(file_info) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                    
                    if not content.strip():
                        continue
                        
                    # Standardize language label
                    if is_dockerfile:
                        lang = "dockerfile"
                    elif is_jenkinsfile or ext == "groovy":
                        lang = "groovy"
                    elif ext in ["sh", "bash"]:
                        lang = "shell"
                    elif ext in ["yml", "yaml"]:
                        lang = "yaml"
                    elif ext in ["j2"]:
                        lang = "jinja"
                    elif ext in ["tf", "tfvars"]:
                        lang = "terraform"
                    elif ext in ["json"]:
                        lang = "json"
                    elif ext in ["sql"]:
                        lang = "sql"
                    elif ext in ["ini", "cfg", "conf"]:
                        lang = "ini"
                    elif ext in ["ps1"]:
                        lang = "powershell"
                    elif ext in ["toml"]:
                        lang = "toml"
                    else:
                        lang = "python"
                    
                    record = {
                        "id": extracted_count,
                        "language": lang,
                        "code": content,
                        "repo_name": repo_name,
                        "path": file_info.filename
                    }
                    
                    output_jsonl_file.write(json.dumps(record) + "\n")
                    extracted_count += 1
                    
                    if extracted_count >= max_files:
                        break
                except Exception as e:
                    # Ignore decode errors or corrupted zip files
                    continue
                    
    print(f"Extracted {extracted_count} files from {repo_name}.")
    return extracted_count

def main():
    output_file = "01_raw_data/hf_datasets/raw_scripts.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Curated massive sources list for a 1-hour deep scraping session.
    # We download directly from GitHub source archives of the largest DevOps and Cloud codebases.
    sources = [
        {
            "url": "https://github.com/bash-it/bash-it/archive/refs/heads/master.zip",
            "repo_name": "bash-it/bash-it",
            "extensions": ["sh", "bash"],
            "max_files": 5000
        },
        {
            "url": "https://github.com/geerlingguy/ansible-for-devops/archive/refs/heads/master.zip",
            "repo_name": "geerlingguy/ansible-for-devops",
            "extensions": ["py", "yml", "yaml"],
            "max_files": 2000
        },
        {
            "url": "https://github.com/ansible/ansible/archive/refs/heads/stable-2.15.zip",
            "repo_name": "ansible/ansible",
            "extensions": ["py", "sh", "yml", "yaml"],
            "max_files": 15000  # Extract up to 15,000 files
        },
        # Category B: Cloud and Container Infrastructure
        {
            "url": "https://github.com/kubernetes-client/python/archive/refs/heads/master.zip",
            "repo_name": "kubernetes-client/python",
            "extensions": ["py"],
            "max_files": 5000
        },
        {
            "url": "https://github.com/GoogleCloudPlatform/google-cloud-python/archive/refs/heads/main.zip",
            "repo_name": "GoogleCloudPlatform/google-cloud-python",
            "extensions": ["py", "sh", "yml", "yaml"],
            "max_files": 8000
        },
        {
            "url": "https://github.com/Azure/azure-cli/archive/refs/heads/main.zip",
            "repo_name": "Azure/azure-cli",
            "extensions": ["py", "sh", "yml", "yaml"],
            "max_files": 8000
        },
        # Category E: Error Handling and Script Robustness
        {
            "url": "https://github.com/kvz/bash3boilerplate/archive/refs/heads/main.zip",
            "repo_name": "kvz/bash3boilerplate",
            "extensions": ["sh", "bash"],
            "max_files": 500
        },
        {
            "url": "https://github.com/ralish/bash-script-template/archive/refs/heads/main.zip",
            "repo_name": "ralish/bash-script-template",
            "extensions": ["sh", "bash"],
            "max_files": 500
        },
        # Category F: Database Automation
        {
            "url": "https://github.com/geerlingguy/ansible-role-postgresql/archive/refs/heads/master.zip",
            "repo_name": "geerlingguy/ansible-role-postgresql",
            "extensions": ["py", "sh", "yml", "yaml", "j2"],
            "max_files": 1000
        },
        {
            "url": "https://github.com/geerlingguy/ansible-role-mysql/archive/refs/heads/master.zip",
            "repo_name": "geerlingguy/ansible-role-mysql",
            "extensions": ["py", "sh", "yml", "yaml", "j2"],
            "max_files": 1000
        },
        {
            "url": "https://github.com/geerlingguy/ansible-role-redis/archive/refs/heads/master.zip",
            "repo_name": "geerlingguy/ansible-role-redis",
            "extensions": ["py", "sh", "yml", "yaml", "j2"],
            "max_files": 1000
        },
        # Docker Compose & Dockerfiles
        {
            "url": "https://github.com/docker/awesome-compose/archive/refs/heads/master.zip",
            "repo_name": "docker/awesome-compose",
            "extensions": ["yml", "yaml", "dockerfile"],
            "max_files": 3000
        },
        # Terraform Infrastructure Configurations & Provider Test Fixtures
        {
            "url": "https://github.com/terraform-aws-modules/terraform-aws-vpc/archive/refs/heads/master.zip",
            "repo_name": "terraform-aws-modules/terraform-aws-vpc",
            "extensions": ["tf", "tfvars"],
            "max_files": 2000
        },
        {
            "url": "https://github.com/hashicorp/terraform-provider-aws/archive/refs/heads/main.zip",
            "repo_name": "hashicorp/terraform-provider-aws",
            "extensions": ["tf", "tfvars"],
            "max_files": 15000  # Extract up to 15,000 files to get rich AWS IAC code
        },
        # Kubernetes Configuration Manifests
        {
            "url": "https://github.com/kubernetes/examples/archive/refs/heads/master.zip",
            "repo_name": "kubernetes/examples",
            "extensions": ["yml", "yaml", "json"],
            "max_files": 3000
        },
        # GitHub Actions Workflows
        {
            "url": "https://github.com/actions/starter-workflows/archive/refs/heads/main.zip",
            "repo_name": "actions/starter-workflows",
            "extensions": ["yml", "yaml"],
            "max_files": 2000
        },
        # Jenkins Pipelines & Examples
        {
            "url": "https://github.com/jenkinsci/pipeline-examples/archive/refs/heads/master.zip",
            "repo_name": "jenkinsci/pipeline-examples",
            "extensions": ["groovy", "jenkinsfile", "yml", "yaml"],
            "max_files": 2000
        },
        # Helm Charts & Kubernetes Templates
        {
            "url": "https://github.com/bitnami/charts/archive/refs/heads/main.zip",
            "repo_name": "bitnami/charts",
            "extensions": ["yml", "yaml", "j2"],
            "max_files": 15000  # Extract up to 15,000 charts
        },
        # Terraform AWS RDS Modules
        {
            "url": "https://github.com/terraform-aws-modules/terraform-aws-rds/archive/refs/heads/master.zip",
            "repo_name": "terraform-aws-modules/terraform-aws-rds",
            "extensions": ["tf", "tfvars"],
            "max_files": 2000
        },
        # Terraform AWS EKS Modules
        {
            "url": "https://github.com/terraform-aws-modules/terraform-aws-eks/archive/refs/heads/master.zip",
            "repo_name": "terraform-aws-modules/terraform-aws-eks",
            "extensions": ["tf", "tfvars"],
            "max_files": 2000
        },
        # GitOps & Declarative ArgoCD Setups
        {
            "url": "https://github.com/argoproj/argo-cd/archive/refs/heads/master.zip",
            "repo_name": "argoproj/argo-cd",
            "extensions": ["yml", "yaml", "sh", "bash"],
            "max_files": 5000
        },
        # Bash Automation Snippets
        {
            "url": "https://github.com/alexanderepstein/Bash-Snippets/archive/refs/heads/master.zip",
            "repo_name": "alexanderepstein/Bash-Snippets",
            "extensions": ["sh", "bash"],
            "max_files": 2000
        },
        # Windows PowerShell Automation Scripts
        {
            "url": "https://github.com/lazywinadmin/PowerShell/archive/refs/heads/master.zip",
            "repo_name": "lazywinadmin/PowerShell",
            "extensions": ["ps1"],
            "max_files": 3000
        },
        # PostgreSQL Cloud Database Automation Operator
        {
            "url": "https://github.com/crunchydata/postgres-operator/archive/refs/heads/master.zip",
            "repo_name": "crunchydata/postgres-operator",
            "extensions": ["yml", "yaml", "sh", "py"],
            "max_files": 3000
        },
        # Prometheus Configurations and Rules
        {
            "url": "https://github.com/prometheus/prometheus/archive/refs/heads/main.zip",
            "repo_name": "prometheus/prometheus",
            "extensions": ["yml", "yaml", "sh", "bash", "json"],
            "max_files": 3000
        }
    ]
    
    # Identify already completed repositories to support resuming
    completed_repos = set()
    if os.path.exists(output_file):
        print(f"Checking existing {output_file} to identify completed repositories...")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            if "repo_name" in record:
                                completed_repos.add(record["repo_name"])
                        except Exception:
                            pass
            print(f"Found {len(completed_repos)} already processed repositories.")
        except Exception as e:
            print(f"Warning: could not parse existing file for resume: {e}")

    total_extracted = 0
    # Open in append mode to preserve previously scraped data
    with open(output_file, "a", encoding="utf-8") as out_f:
        for source in sources:
            repo_name = source["repo_name"]
            if repo_name in completed_repos:
                print(f"Repository {repo_name} already completed. Skipping.")
                continue
                
            count = download_and_extract_zip(
                url=source["url"],
                repo_name=repo_name,
                allowed_extensions=source["extensions"],
                output_jsonl_file=out_f,
                max_files=source.get("max_files", 500)
            )
            total_extracted += count
            
    print(f"\nAll sources downloaded and extracted. Saved {total_extracted} new scripts to {output_file}.")

if __name__ == "__main__":
    main()
