import urllib.request
import json
import time
import os

def get_readme(owner, repo):
    # Fetch directly from raw.githubusercontent.com (no GitHub API used at all)
    common_paths = [
        "main/README.md",
        "master/README.md",
        "main/readme.md",
        "master/readme.md",
        "main/README.markdown",
        "master/README.markdown",
        "main/README.rst",
        "master/README.rst",
        "main/readme.rst",
        "master/readme.rst"
    ]
    
    for path in common_paths:
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{path}"
        req = urllib.request.Request(raw_url)
        req.add_header("User-Agent", "DevOps-Dataset-Builder")
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.getcode() == 200:
                    return response.read().decode('utf-8', errors='ignore')
        except Exception:
            continue
    return None

def main():
    # A curated, top-tier list of DevOps, cloud, database, and robustness repositories.
    # By hardcoding this list, we use ZERO GitHub API calls and bypass all rate limits.
    repos = [
        # Base DevOps & Shell
        ("jlevy", "the-art-of-command-line"),
        ("nvm-sh", "nvm"),
        ("junegunn", "fzf"),
        ("starship", "starship"),
        ("acmesh-official", "acme.sh"),
        ("koalaman", "shellcheck"),
        ("ajeetdsouza", "zoxide"),
        ("alebcay", "awesome-shell"),
        ("mathiasbynens", "dotfiles"),
        ("ansible", "ansible"),
        ("trailofbits", "algo"),
        ("MichaelCade", "90DaysOfDevOps"),
        ("kubernetes-sigs", "kubespray"),
        ("ansible", "awx"),
        ("semaphoreui", "semaphore"),
        ("geerlingguy", "ansible-for-devops"),
        ("moby", "moby"),
        ("traefik", "traefik"),
        ("dani-garcia", "vaultwarden"),
        ("wagoodman", "dive"),
        ("dockur", "windows"),
        ("trimstray", "the-book-of-secret-knowledge"),
        ("nektos", "act"),
        ("go-gitea", "gitea"),
        ("getsentry", "sentry"),
        ("Kong", "kong"),
        ("httpie", "cli"),
        ("dokku", "dokku"),
        ("kestra-io", "kestra"),
        
        # Category B: Cloud & Container Infrastructure
        ("boto", "boto3"),
        ("aws", "aws-cli"),
        ("googleapis", "google-cloud-python"),
        ("Azure", "azure-cli"),
        ("helm", "helm"),
        ("hashicorp", "vault"),
        ("hashicorp", "terraform"),
        ("kubernetes", "kubernetes"),
        ("kubernetes-client", "python"),
        ("kubernetes-sigs", "kustomize"),
        ("argoproj", "argo-cd"),
        ("aws", "aws-cdk"),
        
        # Category E: Error Handling & Robustness
        ("kvz", "bash3boilerplate"),
        ("ralish", "bash-script-template"),
        ("shellspec", "shellspec"),
        ("bats-core", "bats-core"),
        ("kward", "shunit2"),
        
        # Category F: Database Automation
        ("postgres", "postgres"),
        ("major", "MySQLTuner-perl"),
        ("presslabs", "mysql-operator"),
        ("zalando", "postgres-operator"),
        ("redis", "redis"),
        ("mongodb", "mongo"),
        ("flyway", "flyway"),
        ("liquibase", "liquibase"),
        ("ansible-collections", "community.postgresql"),
        ("ansible-collections", "community.mysql")
    ]
    
    output_file = "01_raw_data/github_repos/readmes.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
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

    seen_repos = set()
    count = len(completed_repos)
    
    print(f"Starting direct README downloads for {len(repos)} repositories (No APIs)...")
    
    # Open in append mode to preserve existing data
    with open(output_file, "a", encoding="utf-8") as f:
        for owner, repo in repos:
            full_name = f"{owner}/{repo}"
            if full_name in seen_repos:
                continue
            seen_repos.add(full_name)
            
            if full_name in completed_repos:
                print(f"Repository {full_name} already completed. Skipping.")
                continue
                
            print(f"Scraping README for {full_name}...")
            readme_content = get_readme(owner, repo)
            if readme_content:
                record = {
                    "repo_name": full_name,
                    "stars": 1000, # Curated default stars
                    "url": f"https://github.com/{full_name}",
                    "description": f"Curated DevOps repository: {repo}",
                    "readme_content": readme_content
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                count += 1
                print(f"  Saved. Total: {count}")
            else:
                print(f"  No README found at standard paths. Skipped.")
                
            # Direct raw download is very fast, but let's keep a small politeness sleep
            time.sleep(0.5)
                
    print(f"\nCompleted direct scraping. Saved/verified {count}/{len(repos)} READMEs to {output_file}.")

if __name__ == "__main__":
    main()
