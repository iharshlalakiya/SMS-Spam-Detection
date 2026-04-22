"""Restore all nginx configs - run: sudo python3 automation/restore_nginx.py"""
import subprocess

# Streamlit on port 80
streamlit = """server {
    listen 80;
    server_name _;
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
"""

# MLflow on port 5000
mlflow = """server {
    listen 5000;
    server_name _;
    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host 20.24.104.11:5000;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
        proxy_read_timeout 86400;
    }
}
"""

open("/etc/nginx/sites-available/streamlit", "w").write(streamlit)
open("/etc/nginx/sites-available/mlflow", "w").write(mlflow)
open("/etc/nginx/sites-available/monitor", "w").write("")  # disable 8080

subprocess.run(["ln", "-sf", "/etc/nginx/sites-available/streamlit", "/etc/nginx/sites-enabled/streamlit"])
subprocess.run(["ln", "-sf", "/etc/nginx/sites-available/mlflow",    "/etc/nginx/sites-enabled/mlflow"])
subprocess.run(["rm", "-f", "/etc/nginx/sites-enabled/monitor"])
subprocess.run(["rm", "-f", "/etc/nginx/sites-enabled/default"])

r = subprocess.run(["nginx", "-t"], capture_output=True, text=True)
print(r.stdout, r.stderr)
subprocess.run(["systemctl", "reload", "nginx"])
print("Done!")
print("  Streamlit → http://20.24.104.11")
print("  MLflow    → http://20.24.104.11:5000")
