#!/bin/bash
# Install script for Cat Litter Monitor LaunchAgent services

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "=================================="
echo "Cat Litter Monitor - Service Install"
echo "=================================="
echo ""

# Ensure LaunchAgents directory exists
mkdir -p "$LAUNCH_AGENTS_DIR"

# Copy plist files
echo "📦 Copying service files..."
cp "$SCRIPT_DIR/com.doga.cat-litter-monitor.plist" "$LAUNCH_AGENTS_DIR/"
cp "$SCRIPT_DIR/com.doga.cat-litter-dashboard.plist" "$LAUNCH_AGENTS_DIR/"
echo "✓ Service files copied to $LAUNCH_AGENTS_DIR"
echo ""

# Set correct permissions
chmod 644 "$LAUNCH_AGENTS_DIR/com.doga.cat-litter-monitor.plist"
chmod 644 "$LAUNCH_AGENTS_DIR/com.doga.cat-litter-dashboard.plist"
echo "✓ Permissions set"
echo ""

echo "=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "The services have been installed but NOT started."
echo ""
echo "To start the services:"
echo "  launchctl load ~/Library/LaunchAgents/com.doga.cat-litter-monitor.plist"
echo "  launchctl load ~/Library/LaunchAgents/com.doga.cat-litter-dashboard.plist"
echo ""
echo "To stop the services:"
echo "  launchctl unload ~/Library/LaunchAgents/com.doga.cat-litter-monitor.plist"
echo "  launchctl unload ~/Library/LaunchAgents/com.doga.cat-litter-dashboard.plist"
echo ""
echo "To check service status:"
echo "  launchctl list | grep doga"
echo ""
echo "To view logs:"
echo "  tail -f $SCRIPT_DIR/logs/monitor_stdout.log"
echo "  tail -f $SCRIPT_DIR/logs/monitor_stderr.log"
echo "  tail -f $SCRIPT_DIR/logs/dashboard_stdout.log"
echo "  tail -f $SCRIPT_DIR/logs/dashboard_stderr.log"
echo ""
echo "Once running, access the dashboard at:"
echo "  http://localhost:8080"
echo "  or http://<your-mac-ip>:8080 from another device"
echo ""
