#!/bin/bash
# =============================================================================
# Azure VM Setup — SMS Spam Detection
# Run once on the VM:  bash automation/vm_setup.sh
# =============================================================================
set -e

PROJECT_DIR="/home/azureuser/SMS-Spam-Detection"
VENV="$PROJECT_DIR/.venv"
SERVICE_SRC="$PROJECT_DIR/automation/sms-watcher.service"
SERVICE_DST="/etc/systemd/system/sms-watcher.service"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Azure VM Setup — SMS Spam Detection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. System packages
echo "→ Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3-pip git

# 2. Virtual environment
echo "→ Creating virtual environment..."
python3.11 -m venv "$VENV"
source "$VENV/bin/activate"

# 3. Python dependencies
echo "→ Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt" --quiet

# 4. DVC
echo "→ Verifying DVC..."
dvc --version

# 5. Register sms-watcher service
echo "→ Registering sms-watcher service..."
sudo cp "$SERVICE_SRC" "$SERVICE_DST"

# 6. Register MLflow UI service
echo "→ Registering MLflow UI service..."
sudo cp "$PROJECT_DIR/automation/mlflow.service" /etc/systemd/system/mlflow.service

# 7. Register Streamlit service
echo "→ Registering Streamlit service..."
sudo tee /etc/systemd/system/streamlit.service > /dev/null <<EOF
[Unit]
Description=SMS Spam Detection Streamlit App
After=network.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV/bin/python -m streamlit run app.py --server.port 8501 --server.headless true
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable sms-watcher mlflow streamlit
sudo systemctl start sms-watcher mlflow streamlit

VM_IP=$(curl -s ifconfig.me)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Setup complete!"
echo ""
echo "  Services running:"
echo "    sms-watcher  → auto-retrains when data/raw/*.csv changes"
echo "    streamlit    → http://$VM_IP:8501"
echo "    mlflow       → http://$VM_IP:5000"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status sms-watcher mlflow streamlit"
echo "    tail -f $PROJECT_DIR/logs/watcher.log"
echo "    tail -f $PROJECT_DIR/logs/mlflow.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
