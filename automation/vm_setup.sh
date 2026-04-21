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

# 8. Register data-feed service
echo "→ Registering sms-data-feed service..."
sudo cp "$PROJECT_DIR/automation/sms-data-feed.service" /etc/systemd/system/sms-data-feed.service

# 9. Register monitor dashboard service
echo "→ Registering sms-monitor service..."
sudo cp "$PROJECT_DIR/automation/sms-monitor.service" /etc/systemd/system/sms-monitor.service

sudo systemctl daemon-reload
sudo systemctl enable sms-watcher mlflow streamlit sms-data-feed sms-monitor
sudo systemctl start  sms-watcher mlflow streamlit sms-data-feed sms-monitor

VM_IP=$(curl -s ifconfig.me)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Setup complete!"
echo ""
echo "  Services running:"
echo "    sms-watcher    → auto-retrains when data/raw/*.csv changes"
echo "    sms-data-feed  → continuous data ingestion + pipeline trigger"
echo "    sms-monitor    → real-time HTML dashboard"
echo "    streamlit      → http://$VM_IP:8501"
echo "    mlflow         → http://$VM_IP:5000"
echo "    monitor        → http://$VM_IP:8765"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status sms-watcher sms-data-feed sms-monitor mlflow streamlit"
echo "    tail -f $PROJECT_DIR/logs/data_feed.log"
echo "    tail -f $PROJECT_DIR/logs/monitor.log"
echo "    tail -f $PROJECT_DIR/logs/watcher.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
