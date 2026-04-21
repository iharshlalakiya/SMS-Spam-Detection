#!/bin/bash
# =============================================================================
# Install watcher automation on existing Azure VM
# Run:  bash automation/install_watcher.sh
# =============================================================================
set -e

PROJECT_DIR="/home/azureuser/SMS-Spam-Detection"
WATCHER_SERVICE="/etc/systemd/system/sms-watcher.service"
MLFLOW_SERVICE="/etc/systemd/system/mlflow.service"

echo "→ Pulling latest code (includes watcher files)..."
cd "$PROJECT_DIR"
git fetch origin main
git reset --hard origin/main

echo "→ Detecting Python path..."
PYTHON=$(which python3 || which python)
MLFLOW="$(dirname $PYTHON)/mlflow"
echo "   Using Python: $PYTHON"
echo "   Using MLflow: $MLFLOW"

# Patch watcher service file
echo "→ Installing sms-watcher service..."
sed "s|/home/azureuser/SMS-Spam-Detection/.venv/bin/python|$PYTHON|g" \
    "$PROJECT_DIR/automation/sms-watcher.service" | sudo tee "$WATCHER_SERVICE" > /dev/null

# Patch mlflow service file
echo "→ Installing mlflow service..."
sed "s|/home/azureuser/SMS-Spam-Detection/.venv/bin/mlflow|$MLFLOW|g" \
    "$PROJECT_DIR/automation/mlflow.service" | sudo tee "$MLFLOW_SERVICE" > /dev/null

echo "→ Registering services..."
sudo systemctl daemon-reload
sudo systemctl enable sms-watcher mlflow
sudo systemctl restart sms-watcher mlflow

echo ""
echo "✅ Done! Services are running."
echo ""
echo "   sms-watcher → auto-retrains when data/raw/*.csv changes"
echo "   mlflow      → UI at http://$(curl -s ifconfig.me):5000"
echo ""
echo "   Check status:  sudo systemctl status sms-watcher mlflow"
echo "   Watch logs:    tail -f $PROJECT_DIR/logs/watcher.log"
echo "                  tail -f $PROJECT_DIR/logs/mlflow.log"
