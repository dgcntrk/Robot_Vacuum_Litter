# Cat Litter Monitor - Dashboard & Service Setup

## Overview

The cat litter monitor now includes:
- **Web Dashboard** - Access the live camera feed and status from any device on your network
- **LaunchAgent Services** - Run the monitor headless as a background service on macOS

## Web Dashboard

### Features
- Live camera feed with detection overlays
- Real-time zone status (IDLE, CAT_ENTERED, CAT_INSIDE, etc.)
- Event log showing recent activity
- System stats (uptime, FPS, robot status)
- Clean, dark-themed responsive design

### Running the Dashboard

The dashboard runs independently but reads data from `live_monitor.py`:

```bash
# Start the monitor (this generates the data)
./live_monitor.py

# In another terminal, start the dashboard
./dashboard.py
```

Access the dashboard:
- Local: http://localhost:8080
- Network: http://<your-mac-ip>:8080

### How It Works

1. `live_monitor.py` runs the camera detection and state machine
2. Every second it writes:
   - `logs/latest_frame.jpg` - Annotated camera frame
   - `logs/status.json` - Current system status
3. `dashboard.py` serves these files via a Flask web app
4. The web page auto-refreshes every second

## LaunchAgent Services

Run both the monitor and dashboard as background services that:
- Auto-start on login
- Restart on crash
- Run headless (no GUI windows)
- Log to files

### Installation

```bash
# Install the service files
./install_service.sh
```

This copies the LaunchAgent plists to `~/Library/LaunchAgents/` but does NOT start them.

### Starting the Services

```bash
# Start the monitor (headless, no cv2 windows)
launchctl load ~/Library/LaunchAgents/com.doga.cat-litter-monitor.plist

# Start the dashboard web server
launchctl load ~/Library/LaunchAgents/com.doga.cat-litter-dashboard.plist
```

Both services will now:
- Run automatically on login
- Restart if they crash
- Run in the background

### Stopping the Services

```bash
# Stop the monitor
launchctl unload ~/Library/LaunchAgents/com.doga.cat-litter-monitor.plist

# Stop the dashboard
launchctl unload ~/Library/LaunchAgents/com.doga.cat-litter-dashboard.plist
```

### Checking Status

```bash
# List running services
launchctl list | grep doga

# View live logs
tail -f logs/monitor_stdout.log
tail -f logs/monitor_stderr.log
tail -f logs/dashboard_stdout.log
tail -f logs/dashboard_stderr.log
```

### Service Files

Two LaunchAgent plists are installed:

1. **com.doga.cat-litter-monitor.plist**
   - Runs `live_monitor.py` in headless mode
   - Sets `HEADLESS=true` environment variable
   - Skips cv2 windows, uses sleep() for timing
   - Writes to `logs/monitor_*.log`

2. **com.doga.cat-litter-dashboard.plist**
   - Runs `dashboard.py` web server
   - Listens on port 8080
   - Writes to `logs/dashboard_*.log`

## Headless Mode

When `HEADLESS=true`, `live_monitor.py`:
- Skips `cv2.namedWindow()`, `cv2.imshow()`, `cv2.waitKey()`
- Uses `time.sleep(0.033)` instead of waitKey for timing
- Still runs full detection, state machine, and robot dispatch
- Exports frames and status for dashboard

## File Structure

```
cat-litter-monitor/
├── live_monitor.py                          # Main monitor (modified)
├── dashboard.py                             # Web dashboard (new)
├── install_service.sh                       # Service installer (new)
├── com.doga.cat-litter-monitor.plist       # Monitor service (new)
├── com.doga.cat-litter-dashboard.plist     # Dashboard service (new)
└── logs/
    ├── latest_frame.jpg                     # Latest annotated frame
    ├── status.json                          # Current system status
    ├── events.log                           # Event log
    ├── monitor_stdout.log                   # Monitor service output
    ├── monitor_stderr.log                   # Monitor service errors
    ├── dashboard_stdout.log                 # Dashboard service output
    └── dashboard_stderr.log                 # Dashboard service errors
```

## Usage Scenarios

### Development (with GUI)
```bash
# Run normally with cv2 windows
./live_monitor.py
```

### Testing Dashboard
```bash
# Terminal 1: Run monitor
./live_monitor.py

# Terminal 2: Run dashboard
./dashboard.py

# Browser: http://localhost:8080
```

### Production (background service)
```bash
# Install and start services
./install_service.sh
launchctl load ~/Library/LaunchAgents/com.doga.cat-litter-monitor.plist
launchctl load ~/Library/LaunchAgents/com.doga.cat-litter-dashboard.plist

# Access dashboard from any device
# http://<your-mac-ip>:8080
```

## Troubleshooting

### Dashboard shows "No frame available"
- Check that `live_monitor.py` is running
- Check that `logs/latest_frame.jpg` exists and is recent
- Check monitor logs for errors

### Services won't start
```bash
# Check service status
launchctl list | grep doga

# Manually test the commands
cd /path/to/cat-litter-monitor
HEADLESS=true /opt/homebrew/bin/python3.11 ./live_monitor.py
/opt/homebrew/bin/python3.11 ./dashboard.py
```

### Can't access dashboard from network
- Check firewall settings
- Verify Mac IP address: `ifconfig | grep "inet "`
- Ensure port 8080 is not blocked

## Requirements

- Python 3.11
- Flask (`pip install flask`) - now installed
- All existing dependencies (OpenCV, etc.)

## Security Note

The dashboard runs on `0.0.0.0:8080`, making it accessible to any device on your local network. This is intentional for easy access from phones/tablets, but be aware that anyone on your network can view the camera feed.

To restrict to localhost only, edit `dashboard.py` and change:
```python
app.run(host='0.0.0.0', port=8080, debug=False)
```
to:
```python
app.run(host='127.0.0.1', port=8080, debug=False)
```
