#!/bin/bash
# =============================================================================
# Install watcher automation on existing Azure VM
# Run:  bash automation/install_watcher.sh
# =============================================================================
set -e

PROJECT_DIR="/home/azureuser/SMS-Spam-Detection"
SERVICE_DST="/etc/systemd/system/sms-watcher.service"

echo "→ Pulling latest code (includes watcher files)..."
cd "$PROJECT_DIR"
git fetch origin main
git reset --hard origin/main

echo "→ Detecting Python path..."
PYTHON=$(which python3 || which python)
echo "   Using: $PYTHON"

# Patch the service file with the correct python path on this VM
sed "s|/home/azureuser/SMS-Spam-Detection/.venv/bin/python|$PYTHON|g" \
    "$PROJECT_DIR/automation/sms-watcher.service" | sudo tee "$SERVICE_DST" > /dev/null

echo "→ Registering sms-watcher systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable sms-watcher
sudo systemctl restart sms-watcher

echo ""
echo "✅ Done! Watcher is running."
echo ""
echo "   Drop a new CSV into:  $PROJECT_DIR/data/raw/"
echo "   Pipeline auto-runs within 30 seconds."
echo ""
echo "   Check status:  sudo systemctl status sms-watcher"
echo "   Watch logs:    tail -f $PROJECT_DIR/logs/watcher.log"
