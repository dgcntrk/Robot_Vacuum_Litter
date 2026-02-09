#!/usr/bin/env python3
"""
Cat Litter Monitor - Web Dashboard
Simple web interface showing live camera feed with overlays and status panel.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, Response, jsonify, send_file

app = Flask(__name__)

# Paths
LOGS_DIR = Path(__file__).parent / "logs"
STATUS_FILE = LOGS_DIR / "status.json"
FRAME_FILE = LOGS_DIR / "latest_frame.jpg"

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)


def load_status():
    """Load status from JSON file."""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading status: {e}")
    
    # Return default status if file doesn't exist or error
    return {
        'timestamp': time.time(),
        'zones': [],
        'events': [],
        'uptime': 0,
        'robot_enabled': False,
        'fps': 0.0,
    }


def format_uptime(seconds):
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def format_timestamp(timestamp):
    """Format timestamp as readable time."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%H:%M:%S")


@app.route('/')
def index():
    """Main dashboard page."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cat Litter Monitor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #30363d;
        }
        
        h1 {
            font-size: 2.5em;
            color: #58a6ff;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #8b949e;
            font-size: 1.1em;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 1200px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .video-panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }
        
        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 4px;
            overflow: hidden;
        }
        
        #camera-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .status-panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }
        
        .status-section {
            margin-bottom: 25px;
        }
        
        .status-section:last-child {
            margin-bottom: 0;
        }
        
        .status-section h2 {
            font-size: 1.3em;
            color: #58a6ff;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #30363d;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #21262d;
        }
        
        .stat-row:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            color: #8b949e;
        }
        
        .stat-value {
            color: #c9d1d9;
            font-weight: 600;
        }
        
        .zone-card {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 10px;
        }
        
        .zone-card:last-child {
            margin-bottom: 0;
        }
        
        .zone-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .zone-name {
            font-weight: 600;
            color: #58a6ff;
        }
        
        .zone-state {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .state-idle { background: #30363d; color: #8b949e; }
        .state-cat_entered { background: #ffd60a; color: #000; }
        .state-cat_inside { background: #da3633; color: #fff; }
        .state-cat_exited { background: #fb8500; color: #fff; }
        .state-cooldown { background: #1f6feb; color: #fff; }
        .state-dispatch_ready { background: #2ea043; color: #fff; }
        
        .zone-details {
            font-size: 0.9em;
            color: #8b949e;
        }
        
        .event-log {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }
        
        .event-log h2 {
            font-size: 1.3em;
            color: #58a6ff;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #30363d;
        }
        
        .event-item {
            padding: 8px 0;
            border-bottom: 1px solid #21262d;
            font-size: 0.9em;
        }
        
        .event-item:last-child {
            border-bottom: none;
        }
        
        .event-time {
            color: #8b949e;
            margin-right: 10px;
        }
        
        .event-level {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 8px;
        }
        
        .level-INFO { background: #30363d; color: #8b949e; }
        .level-STATE { background: #ffd60a; color: #000; }
        .level-ENTER { background: #2ea043; color: #fff; }
        .level-EXIT { background: #fb8500; color: #fff; }
        .level-DISPATCH { background: #2ea043; color: #fff; }
        .level-ROBOT { background: #1f6feb; color: #fff; }
        .level-SYSTEM { background: #6e7681; color: #fff; }
        
        .event-message {
            color: #c9d1d9;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }
        
        .status-online {
            background: #2ea043;
        }
        
        .status-offline {
            background: #da3633;
        }
        
        .no-data {
            text-align: center;
            color: #8b949e;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🐱 Cat Litter Monitor</h1>
            <div class="subtitle">Live monitoring with automatic robot dispatch</div>
        </header>
        
        <div class="main-content">
            <div class="video-panel">
                <h2 style="margin-bottom: 15px; color: #58a6ff;">Live Camera Feed</h2>
                <div class="video-container">
                    <img id="camera-feed" src="/frame" alt="Camera Feed">
                </div>
            </div>
            
            <div class="status-panel">
                <div class="status-section">
                    <h2>System Status</h2>
                    <div class="stat-row">
                        <span class="stat-label">Status</span>
                        <span class="stat-value">
                            <span class="status-indicator status-online"></span>
                            Online
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Uptime</span>
                        <span class="stat-value" id="uptime">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">FPS</span>
                        <span class="stat-value" id="fps">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Robot</span>
                        <span class="stat-value" id="robot-status">--</span>
                    </div>
                </div>
                
                <div class="status-section">
                    <h2>Zone Status</h2>
                    <div id="zones-container">
                        <div class="no-data">No zones detected</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="event-log">
            <h2>Recent Events</h2>
            <div id="events-container">
                <div class="no-data">No events yet</div>
            </div>
        </div>
    </div>
    
    <script>
        // Refresh camera feed
        let frameNumber = 0;
        setInterval(() => {
            const img = document.getElementById('camera-feed');
            frameNumber++;
            img.src = '/frame?t=' + frameNumber;
        }, 1000);
        
        // Update status
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update system stats
                    document.getElementById('uptime').textContent = formatUptime(data.uptime || 0);
                    document.getElementById('fps').textContent = (data.fps || 0).toFixed(1);
                    document.getElementById('robot-status').innerHTML = data.robot_enabled 
                        ? '<span class="status-indicator status-online"></span>Enabled'
                        : '<span class="status-indicator status-offline"></span>Disabled';
                    
                    // Update zones
                    const zonesContainer = document.getElementById('zones-container');
                    if (data.zones && data.zones.length > 0) {
                        zonesContainer.innerHTML = data.zones.map(zone => {
                            const state = zone.state || 'idle';
                            const duration = zone.state_duration_seconds || 0;
                            let stateText = state.toUpperCase();
                            
                            // Add countdown info
                            if (state === 'cooldown' && zone.cooldown_remaining !== undefined) {
                                stateText += ` (${zone.cooldown_remaining.toFixed(1)}s)`;
                            } else if (state === 'cat_entered' && zone.time_to_confirmation !== undefined) {
                                stateText += ` (${zone.time_to_confirmation.toFixed(1)}s)`;
                            } else {
                                stateText += ` (${duration.toFixed(1)}s)`;
                            }
                            
                            return `
                                <div class="zone-card">
                                    <div class="zone-header">
                                        <span class="zone-name">${zone.zone_name || 'Unknown'}</span>
                                        <span class="zone-state state-${state}">${stateText}</span>
                                    </div>
                                    <div class="zone-details">
                                        Total sessions: ${zone.total_sessions || 0}
                                    </div>
                                </div>
                            `;
                        }).join('');
                    } else {
                        zonesContainer.innerHTML = '<div class="no-data">No zones detected</div>';
                    }
                    
                    // Update events
                    const eventsContainer = document.getElementById('events-container');
                    if (data.events && data.events.length > 0) {
                        eventsContainer.innerHTML = data.events.slice().reverse().map(event => {
                            const level = event.level || 'INFO';
                            return `
                                <div class="event-item">
                                    <span class="event-time">${event.timestamp.split(' ')[1]}</span>
                                    <span class="event-level level-${level}">${level}</span>
                                    <span class="event-message">${event.message}</span>
                                </div>
                            `;
                        }).join('');
                    } else {
                        eventsContainer.innerHTML = '<div class="no-data">No events yet</div>';
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }
        
        function formatUptime(seconds) {
            if (seconds < 60) {
                return Math.floor(seconds) + 's';
            } else if (seconds < 3600) {
                const minutes = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return minutes + 'm ' + secs + 's';
            } else {
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                return hours + 'h ' + minutes + 'm';
            }
        }
        
        // Update every 1 second
        updateStatus();
        setInterval(updateStatus, 1000);
    </script>
</body>
</html>
"""
    return html


@app.route('/frame')
def get_frame():
    """Serve the latest camera frame."""
    try:
        if FRAME_FILE.exists():
            return send_file(FRAME_FILE, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error serving frame: {e}")
    
    # Return placeholder if no frame available
    from flask import make_response
    return make_response("No frame available", 404)


@app.route('/api/status')
def get_status():
    """API endpoint for status data."""
    status = load_status()
    return jsonify(status)


def main():
    """Run the dashboard server."""
    print("=" * 60)
    print("Cat Litter Monitor - Web Dashboard")
    print("=" * 60)
    print("")
    print("Starting server on http://0.0.0.0:8080")
    print("Access from any device on your network")
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == '__main__':
    main()
