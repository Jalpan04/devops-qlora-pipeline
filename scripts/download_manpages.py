import urllib.request
import os
import time

def download_manpage(command):
    url = f"https://man.archlinux.org/man/{command}"
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "DevOps-Dataset-Builder")
        try:
            # Get redirect URL to know the exact section
            with urllib.request.urlopen(req, timeout=10) as response:
                redirect_url = response.geturl()
                
            txt_url = f"{redirect_url}.txt"
            
            txt_req = urllib.request.Request(txt_url)
            txt_req.add_header("User-Agent", "DevOps-Dataset-Builder")
            with urllib.request.urlopen(txt_req, timeout=10) as txt_response:
                return txt_response.read().decode('utf-8', errors='ignore')
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # 404 is permanent, don't retry
                print(f"Error downloading manpage for {command}: HTTP Error 404: Not Found")
                return None
            print(f"Attempt {attempt}/{max_retries} failed for {command}: HTTP Error {e.code}")
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed for {command}: {e}")
            
        if attempt < max_retries:
            time.sleep(2)
            
    print(f"Failed to fetch manpage for {command} after {max_retries} attempts.")
    return None

def main():
    # Expanded list of commands to include a comprehensive set of Linux utilities and development tools
    commands = [
        # Text Processing & Utilities
        "grep", "awk", "sed", "jq", "cat", "less", "more", "tail", "head", "wc",
        "diff", "patch", "sort", "uniq", "tee", "cut", "tr", "split", "xargs",
        
        # Filesystem & Storage
        "ls", "find", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "ln", "pwd",
        "chmod", "chown", "df", "du", "dd", "mount", "umount", "fdisk", "parted", "mkfs",
        
        # Network & Connectivity
        "curl", "wget", "ssh", "ssh-keygen", "ssh-copy-id", "scp", "sftp", "rsync",
        "ping", "traceroute", "nslookup", "dig", "host", "ifconfig", "ip", "route",
        "ss", "netstat", "tcpdump", "nc", "nmap",
        
        # Security & Firewall
        "iptables", "ufw", "firewalld", "su", "sudo", "passwd",
        
        # System & Process Management
        "systemctl", "systemd", "journalctl", "dmesg", "top", "htop", "free",
        "ps", "kill", "killall", "pkill", "pgrep", "nohup", "screen", "tmux",
        "lsof", "cron", "crontab", "uptime", "uname", "hostname", "whoami", "id", "groups",
        
        # Package Management
        "apt", "apt-get", "yum", "dnf", "dpkg", "rpm", "pip", "npm", "yarn", "cargo",
        
        # Development, Containers & Orchestration
        "docker", "docker-compose", "git", "make", "ansible", "ansible-playbook",
        "python", "perl", "ruby", "gcc", "g++", "clang", "ld", "nm", "objdump",
        "strace", "ltrace", "chroot",
        
        # Cloud & Container CLI tools
        "aws", "gcloud", "az", "helm", "vault",
        
        # Database CLI tools
        "psql", "pg_dump", "mysql", "mysqldump", "redis-cli", "mongosh", "mysqladmin", "pg_ctl",
        
        # Shell Builtins & Miscellaneous
        "env", "export", "history", "date", "sleep", "who", "w", "last"
    ]
    
    output_dir = "01_raw_data/linux_manpages"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    skipped_count = 0
    for cmd in commands:
        out_path = os.path.join(output_dir, f"{cmd}.txt")
        if os.path.exists(out_path):
            skipped_count += 1
            success_count += 1
            continue
            
        print(f"Fetching manpage for {cmd}...")
        content = download_manpage(cmd)
        if content:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
            success_count += 1
        else:
            print(f"Failed to fetch manpage for {cmd}")
            
        time.sleep(1.0) # Politeness delay
        
    print(f"Completed download of manpages. Success: {success_count}/{len(commands)} (Skipped: {skipped_count})")

if __name__ == "__main__":
    main()
