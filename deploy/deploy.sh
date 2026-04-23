#!/usr/bin/env bash
# One-shot deployer for Choreo Studio alongside an existing Choreo (Dance).
#
# Expected runtime layout on the target box:
#   /opt/choreo/         <- existing Dance app (port 5001), untouched
#   /opt/choreo-studio/  <- this app (port 5002), managed here
#   /etc/nginx/sites-*   <- single combined site routing both
#
# Usage:
#   sudo SRC_DIR=/root/ChoreoStudio bash deploy.sh
set -euo pipefail

SRC_DIR="${SRC_DIR:-$(pwd)}"
TARGET=/opt/choreo-studio

echo "==> Source: $SRC_DIR"
echo "==> Target: $TARGET"

# 1. apt packages (opencv deps, ffmpeg, nginx, python)
echo "==> Installing system packages"
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-pip \
    ffmpeg \
    nginx \
    curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1

# 2. user (reuse existing `choreo` user if present)
if ! id choreo >/dev/null 2>&1; then
    echo "==> Creating choreo user"
    useradd -r -m -d /opt/choreo -s /bin/false choreo
fi

# 3. app files
echo "==> Installing app files to $TARGET"
mkdir -p "$TARGET/jobs" "$TARGET/static" "$TARGET/deploy" "$TARGET/logs"
if command -v rsync >/dev/null; then
    rsync -a --delete \
        --exclude 'jobs/' --exclude 'studio.db*' --exclude '.venv/' --exclude 'venv/' \
        --exclude '__pycache__/' --exclude '*.pyc' \
        --exclude 'logs/' --exclude '*.log' --exclude '.DS_Store' \
        "$SRC_DIR"/ "$TARGET"/
else
    cp -R "$SRC_DIR"/. "$TARGET"/
fi
mkdir -p "$TARGET/jobs" "$TARGET/static" "$TARGET/deploy" "$TARGET/logs"

# 4. python venv
echo "==> Creating virtualenv"
python3 -m venv "$TARGET/venv"
"$TARGET/venv/bin/pip" install --upgrade pip wheel
"$TARGET/venv/bin/pip" install -r "$TARGET/requirements.txt"

# 5. ownership
chown -R choreo:choreo "$TARGET"

# 6. systemd
echo "==> Installing systemd unit"
install -m 644 "$TARGET/deploy/choreo-studio.service" /etc/systemd/system/choreo-studio.service
systemctl daemon-reload
systemctl enable choreo-studio
systemctl restart choreo-studio

# 7. nginx — single combined site for both apps
echo "==> Installing combined nginx site"
# Remove the old choreo site if it exists (Studio config supersedes it).
rm -f /etc/nginx/sites-enabled/choreo /etc/nginx/sites-available/choreo
install -m 644 "$TARGET/deploy/nginx-combined.conf" /etc/nginx/sites-available/choreo-combined
ln -sf /etc/nginx/sites-available/choreo-combined /etc/nginx/sites-enabled/choreo-combined
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

# 8. status
echo
echo "==> Done."
systemctl --no-pager -l status choreo-studio | head -15
echo
IP=$(curl -s ifconfig.me || echo 'YOUR-IP')
echo "==> Studio:       http://$IP/"
echo "==> Dance legacy: http://$IP/dance/"
echo "==> Logs:         journalctl -u choreo-studio -f"
