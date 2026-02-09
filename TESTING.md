# Testing the Dashboard and Services

## Quick Start Test

### 1. Test the Dashboard (without live camera)

The dashboard is ready to test immediately:

```bash
cd /path/to/cat-litter-monitor

# Start the dashboard
./dashboard.py
```

Open in your browser: http://localhost:8080

You should see:
- ✓ Dashboard page loads
- ✓ Placeholder camera frame displays
- ✓ System status shows (uptime, FPS, robot status)
- ✓ "No zones detected" message
- ✓ "No events yet" message

**Press Ctrl+C to stop**

### 2. Test with Live Monitor

In one terminal:
```bash
cd /path/to/cat-litter-monitor
./live_monitor.py
```

In another terminal:
```bash
cd /path/to/cat-litter-monitor
./dashboard.py
```

Now the dashboard should show:
- ✓ Live camera feed with overlays
- ✓ Detected litter box zones
- ✓ Real-time state updates
- ✓ Event log with actual events
- ✓ Live FPS counter

### 3. Verify Dashboard Export

While `live_monitor.py` is running, check that it's writing data:

```bash
# Check that status updates every second
watch -n 1 cat logs/status.json

# Check frame timestamp
ls -lh logs/latest_frame.jpg

# Watch the event log
tail -f logs/events.log
```

You should see:
- ✓ `status.json` updates every second
- ✓ `latest_frame.jpg` timestamp updates every second
- ✓ Events appear in `events.log`

### 4. Test Headless Mode

Test that the monitor works without GUI:

```bash
HEADLESS=true ./live_monitor.py
```

You should see:
- ✓ No cv2 windows appear
- ✓ Monitor runs and logs events
- ✓ `logs/status.json` updates
- ✓ `logs/latest_frame.jpg` updates
- ✓ Dashboard works normally

**Press Ctrl+C to stop**

### 5. Test LaunchAgent Installation

```bash
# Install the service files
./install_service.sh
```

You should see:
- ✓ "Installation complete!" message
- ✓ Instructions for starting/stopping services
- ✓ Files copied to ~/Library/LaunchAgents/

Verify the files:
```bash
ls -l ~/Library/LaunchAgents/com.doga.cat-litter*.plist
```

### 6. Test Services (Optional)

**WARNING**: This will start the monitor and dashboard as background services.

```bash
# Load the monitor service
launchctl load ~/Library/LaunchAgents/com.doga.cat-litter-monitor.plist

# Load the dashboard service
launchctl load ~/Library/LaunchAgents/com.doga.cat-litter-dashboard.plist

# Check they're running
launchctl list | grep doga
```

You should see both services listed with PIDs.

Access the dashboard: http://localhost:8080

Check the logs:
```bash
tail -f logs/monitor_stdout.log
tail -f logs/dashboard_stdout.log
```

**To stop the services:**
```bash
launchctl unload ~/Library/LaunchAgents/com.doga.cat-litter-monitor.plist
launchctl unload ~/Library/LaunchAgents/com.doga.cat-litter-dashboard.plist
```

## Network Access Test

Find your Mac's IP address:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Access the dashboard from another device (phone, tablet, laptop) on the same network:
```
http://<your-mac-ip>:8080
```

You should see the full dashboard with live camera feed.

## Expected File Structure

After running the tests, you should have:

```
cat-litter-monitor/
├── live_monitor.py                    ✓ Modified with HEADLESS support
├── dashboard.py                       ✓ New Flask web app
├── install_service.sh                 ✓ Service installer script
├── com.doga.cat-litter-monitor.plist ✓ Monitor LaunchAgent
├── com.doga.cat-litter-dashboard.plist ✓ Dashboard LaunchAgent
├── DASHBOARD_README.md                ✓ Documentation
├── TESTING.md                         ✓ This file
└── logs/
    ├── latest_frame.jpg               ✓ Updated every second
    ├── status.json                    ✓ Updated every second
    ├── events.log                     ✓ Event history
    ├── monitor_stdout.log             ✓ Service output (when running as service)
    ├── monitor_stderr.log             ✓ Service errors
    ├── dashboard_stdout.log           ✓ Dashboard output
    └── dashboard_stderr.log           ✓ Dashboard errors
```

## Troubleshooting

### Dashboard shows old frame
- Stop and restart `live_monitor.py`
- Check file timestamp: `ls -lh logs/latest_frame.jpg`

### Dashboard shows 404 for frame
- Check that `logs/latest_frame.jpg` exists
- Run `live_monitor.py` first to generate frames

### Services won't start
```bash
# Check for errors in the plist
plutil -lint ~/Library/LaunchAgents/com.doga.cat-litter-monitor.plist

# Try running manually
cd /path/to/cat-litter-monitor
HEADLESS=true /opt/homebrew/bin/python3.11 ./live_monitor.py
```

### Port 8080 already in use
```bash
# Find what's using port 8080
lsof -i :8080

# Kill the process or change the port in dashboard.py
```

## Success Criteria

✅ All modifications complete:
- `live_monitor.py` modified to export status and frames
- `live_monitor.py` supports HEADLESS mode
- `dashboard.py` created and working
- LaunchAgent plists created
- Installation script created
- Documentation complete

✅ Dashboard works:
- Displays live camera feed
- Shows zone status
- Shows event log
- Updates every second
- Accessible from network

✅ Services work:
- Monitor runs headless
- Dashboard serves web interface
- Both restart on crash
- Logs are captured

✅ No breaking changes:
- Original `live_monitor.py` functionality preserved
- cv2 windows still work when not headless
- All detection and state machine logic intact
