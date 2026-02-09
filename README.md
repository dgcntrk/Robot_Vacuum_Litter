# 🤖🐱 Robot Vacuum Litter

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/platform-macos-lightgrey.svg)](https://www.apple.com/macos/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-CoreML-green.svg)](https://docs.ultralytics.com/)

> **AI-powered pet care automation.** Watch your cat's litter box with a camera, detect usage in real-time, and automatically dispatch a **Shark Matrix robot vacuum** to clean up afterward — all without lifting a finger.

![Architecture](https://img.shields.io/badge/Apple%20Silicon-Optimized-FF6B6B?style=flat-square&logo=apple)
![CoreML](https://img.shields.io/badge/CoreML-Neural%20Engine-4ECDC4?style=flat-square)
![RTSP](https://img.shields.io/badge/RTSP-Camera-45B7D1?style=flat-square)

---

## 📖 Overview

**Robot Vacuum Litter** is an intelligent monitoring system that bridges computer vision and smart home automation. Using Apple Silicon-optimized YOLOv8 detection running on CoreML, it watches your cat's litter box, tracks their visits, and automatically sends your Shark robot vacuum to clean the area once it's safe to do so.

### How It Works

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   RTSP      │────▶│  YOLOv8 CoreML  │────▶│   State Machine │
│   Camera    │     │  Cat Detection  │     │   (6 States)    │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                     │
                         ┌───────────────────────────┘
                         ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Shark     │◀────│   Stakra API    │◀────│  Dispatch Ready │
│   Matrix    │     │  (Auth0 + HMAC) │     │   Trigger       │
│   Vacuum    │     └─────────────────┘     └─────────────────┘
└─────────────┘
```

1. **Camera captures** the litter box area via RTSP
2. **YOLOv8 on CoreML** detects cats in real-time (<500ms latency)
3. **Dynamic litter box detection** automatically locates the box (no manual zoning!)
4. **State machine** tracks: `IDLE → CAT_ENTERED → CAT_INSIDE → CAT_EXITED → COOLDOWN → DISPATCH_READY`
5. **Shark API** dispatches the robot to the configured room
6. **Emergency stop** triggers if the cat returns while cleaning

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Real-time Detection** | YOLOv8 running on Apple Neural Engine (CoreML) |
| 📦 **Auto Zone Detection** | Apple Vision framework automatically finds your litter box |
| 🎯 **State Machine** | Intelligent occupancy tracking with grace periods & cooldowns |
| 🤖 **Shark Integration** | Full API control: dispatch, stop, return to dock |
| 🛡️ **Safety First** | Emergency stop if cat returns during cleaning |
| 📊 **Web Dashboard** | Live feed, stats, and real-time state visualization |
| 🍎 **Apple Silicon** | Optimized for M1/M2/M3 Macs using CoreML |
| 🔧 **Configurable** | YAML-based config with environment overrides |

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph Detection["Detection Pipeline"]
        A[RTSP Camera] --> B[Frame Capture]
        B --> C[YOLOv8 CoreML]
        C --> D[Cat Detection]
        B --> E[Apple Vision]
        E --> F[Litter Box Detection]
    end
    
    subgraph State["State Management"]
        D --> G{Zone Overlap?}
        F --> G
        G --> H[Zone State Machine]
        H --> I[IDLE]
        H --> J[CAT_ENTERED]
        H --> K[CAT_INSIDE]
        H --> L[CAT_EXITED]
        H --> M[COOLDOWN]
        H --> N[DISPATCH_READY]
    end
    
    subgraph Robot["Robot Control"]
        N --> O[Shark Controller]
        O --> P[Auth0 Auth]
        P --> Q[Stakra API]
        Q --> R[Shark Matrix]
        K -. Emergency .-> S[Emergency Stop]
        S --> R
    end
    
    subgraph Dashboard["Monitoring"]
        H --> T[Web Dashboard]
        T --> U[MJPEG Stream]
        T --> V[WebSocket Events]
        T --> W[REST API]
    end
```

---

## 🤖 Shark Robot Integration

The **star of the show** — direct control of your Shark Matrix robot vacuum through reverse-engineered APIs.

### Authentication Flow

```python
# 1. Auth0 Refresh Token → ID Token
POST https://login.sharkninja.com/oauth/token
{
    "grant_type": "refresh_token",
    "client_id": "YOUR_CLIENT_ID",
    "refresh_token": "YOUR_REFRESH_TOKEN"
}

# 2. HMAC-Signed Requests to Stakra API
# Every request is signed with AWS Signature V4-style HMAC-SHA256
# using your API key and the current timestamp
```

### API Capabilities

| Command | API Call | Description |
|---------|----------|-------------|
| **Dispatch** | `PATCH /devices/{household}/{dsn}` | Send robot to specific room or whole house |
| **Stop** | `PATCH` with `Operating_Mode: 0` | Emergency stop |
| **Return** | `PATCH` with `Operating_Mode: 4` | Return to dock |
| **Status** | `GET /devices/{household}/{dsn}` | Get robot state, battery, room list |

### Room-Specific Cleaning

```python
# Configure room-specific dispatch
areas_payload = {
    "areas_to_clean": {"UserRoom": ["Litter"]},
    "clean_count": 1,
    "floor_id": "floor_1",
    "cleantype": "dry"
}

# Send command via device shadow
await client.set_desired_properties({
    "AreasToClean_V3": json.dumps(areas_payload),
    "Operating_Mode": 2  # START
})
```

### State Machine Integration

The robot is dispatched only when the state machine reaches `DISPATCH_READY`:

```
Cat enters ──► CAT_ENTERED ──► CAT_INSIDE ──► Cat exits ──► CAT_EXITED
     │              │              │                               │
     │              │              │                               │
     └──────────────┴──────────────┘                               │
                    (5s minimum occupancy)                        │
                                                                  ▼
                    ┌───────────────────────────────────────► COOLDOWN
                    │                                          (60s)
                    │                                              │
                    │                                              ▼
                    └──────────────────────────────────────► DISPATCH_READY
                                                                  │
                                                                  ▼
                                                          ┌───────────────┐
                                                          │ Shark.dispatch │
                                                          │  (room="Litter")│
                                                          └───────────────┘
```

### Emergency Stop

If a cat is detected while the robot is cleaning:

```python
async def on_cat_entered(session):
    if not robot.is_available():  # Robot is cleaning
        logger.warning("Cat detected while cleaning! Emergency stop!")
        await robot.stop()
```

---

## 🚀 Quick Start

### Prerequisites

- macOS 12+ with Apple Silicon (M1/M2/M3)
- Python 3.9+
- RTSP-capable camera (WiFi IP camera)
- Shark Matrix robot vacuum

### 1. Clone & Install

```bash
git clone https://github.com/dgcntrk/Robot_Vacuum_Litter.git
cd Robot_Vacuum_Litter
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Camera Setup (RTSP)

Configure your IP camera and get the RTSP URL:

```yaml
# config/settings.yaml
camera:
  rtsp_url: "rtsp://admin:password@192.168.1.100:554/live0"
  fps: 15
  resolution: [640, 480]
  use_tcp: true
```

**Common RTSP URL formats:**
- Reolink: `rtsp://admin:PASSWORD@IP:554/h264Preview_01_main`
- Hikvision: `rtsp://admin:PASSWORD@IP:554/Streaming/Channels/101`
- Generic: `rtsp://USER:PASS@IP:554/live`

### 3. Shark Robot Configuration

#### 🔐 Capturing Your Shark Credentials

This is the most complex part of the setup, but once done, everything works automatically. The Shark app uses **Auth0** for authentication and the **Stakra API** for robot control. You'll need to intercept the app's network traffic to extract the required credentials.

##### Required Credentials

| Credential | Description | Where It Appears |
|------------|-------------|------------------|
| **Auth0 Refresh Token** | Long-lived token for Auth0 authentication | Auth0 token response body (`refresh_token`) |
| **Auth0 Client ID** | Your app's client identifier | Auth0 token request body (`client_id`) |
| **Stakra API Key** | API key for Stakra requests | Stakra API request headers (`x-api-key`) |
| **Household ID** | Your home's unique identifier | Device list API response (`household_id`) |
| **DSN** | Device Serial Number (your robot) | Device list API response (`dsn`) |
| **Floor ID** | Your floor plan's ID | Floor plan API response |
| **Room Name** | The room you want to clean | Floor plan API response (`rooms` list) |

> 💡 **Note:** The refresh token auto-rotates on each use — the code handles this automatically and updates `config/.shark_token` for you.

##### Method: Network Traffic Interception

You'll need to intercept HTTPS traffic from the Shark/SharkClean mobile app. We recommend **mitmproxy** (free, open-source) or **Charles Proxy** (paid, easier GUI).

###### Step 1: Install mitmproxy (Recommended)

```bash
# macOS with Homebrew
brew install mitmproxy

# Or download from https://mitmproxy.org/
```

###### Step 2: Start the Proxy

```bash
# Terminal 1: Start mitmproxy with verbose logging
mitmproxy --mode regular --listen-port 8080

# Or for simpler console output:
# mitmdump --mode regular --listen-port 8080
```

###### Step 3: Configure Your Phone

1. **Connect your phone to the same WiFi network as your computer**
2. **Set up the proxy on your phone:**
   - **iOS:** Settings → Wi-Fi → (i) next to your network → Configure Proxy → Manual
     - Server: Your computer's IP address (e.g., `192.168.1.50`)
     - Port: `8080`
   - **Android:** Settings → Wi-Fi → Long press your network → Modify → Advanced → Proxy → Manual
     - Hostname: Your computer's IP address
     - Port: `8080`

3. **Install the CA certificate on your phone:**
   - Open Safari/Chrome on your phone
   - Navigate to `http://mitm.it`
   - Tap **Get mitmproxy-ca-cert** for your platform
   - Follow the installation prompts:
     - **iOS:** Install → Enter passcode → Trust certificate in Settings → General → About → Certificate Trust Settings
     - **Android:** Install → Name it "mitmproxy" → VPN and apps

###### Step 4: Capture the Credentials

Open the **Shark** or **SharkClean** app on your phone and:
1. Log out if already logged in
2. Log back in with your credentials
3. Navigate through your home/rooms
4. Watch the mitmproxy window for intercepted requests

###### Step 5: Extract Each Credential

**🔑 Auth0 Refresh Token & Client ID**

Look for a POST request to:
```
https://login.sharkninja.com/oauth/token
```

In the **request body**, you'll find:
```json
{
  "grant_type": "refresh_token",
  "client_id": "YOUR_AUTH0_CLIENT_ID",     ← Copy this
  "refresh_token": "YOUR_REFRESH_TOKEN"     ← Copy this
}
```

In the **response body**, you'll see:
```json
{
  "access_token": "...",
  "id_token": "...",
  "refresh_token": "YOUR_NEW_REFRESH_TOKEN",  ← Also save this
  "expires_in": 86400
}
```

**🔑 Stakra API Key**

Look for requests to:
```
https://stakra.slatra.thor.skegox.com/...
```

In the **request headers**, find:
```
x-api-key: YOUR_STAKRA_API_KEY    ← Copy this
```

**🔑 Household ID & DSN**

Look for a GET request to:
```
https://stakra.slatra.thor.skegox.com/devicesEndUserController/{household_id}/devices
```

Or a response containing your devices:
```json
{
  "devices": [
    {
      "dsn": "YOUR_DSN_HERE",              ← Copy this
      "model": "Shark Matrix",
      "name": "My Robot"
    }
  ],
  "household_id": "YOUR_HOUSEHOLD_ID"      ← Copy this
}
```

**🔑 Floor ID & Room Names**

Look for a request/response related to floor plans:
```
https://stakra.slatra.thor.skegox.com/.../floorplan
```

In the response, you'll find:
```json
{
  "floor_id": "YOUR_FLOOR_ID",           ← Copy this
  "rooms": [
    {"id": "room_1", "name": "Kitchen"},
    {"id": "room_2", "name": "Living Room"},
    {"id": "room_3", "name": "Litter"}   ← Your target room name
  ]
}
```

###### Step 6: Save Your Credentials

Create the token file:
```bash
echo '{"auth0_refresh_token": "YOUR_REFRESH_TOKEN"}' > config/.shark_token
chmod 600 config/.shark_token  # Keep it secure
```

Update your `config/settings.yaml`:
```yaml
robot:
  enabled: true
  room_name: "Litter"           # Exact room name from the app
  type: "shark"
  household_id: "YOUR_HOUSEHOLD_ID"
  dsn: "YOUR_DSN"
  floor_id: "YOUR_FLOOR_ID"     # Optional, auto-detected if not set
  dispatch_delay_seconds: 5
  emergency_stop_on_cat_detected: true
```

Edit `src/robot/shark.py` and replace the placeholder constants:
```python
AUTH0_CLIENT_ID = "YOUR_AUTH0_CLIENT_ID"
STAKRA_API_KEY = "YOUR_STAKRA_API_KEY"
```

##### Alternative: Charles Proxy (GUI)

If you prefer a graphical interface:
1. Download [Charles Proxy](https://www.charlesproxy.com/) (free trial)
2. Install and launch Charles
3. Enable SSL Proxying: Proxy → SSL Proxying Settings → Add `*.sharkninja.com` and `*.slatra.thor.skegox.com`
4. Install the Charles certificate on your phone (Help → SSL Proxying → Install Charles Root Certificate on a Mobile Device)
5. Follow the same traffic capture steps above
6. Look for the same requests in Charles' request/response viewer

##### Troubleshooting Credential Capture

**App won't load / SSL errors?**
- Make sure the CA certificate is properly trusted on your phone
- On iOS, you must enable the certificate in Settings → General → About → Certificate Trust Settings
- Some apps use certificate pinning — the Shark app does not, so this should work

**Can't find the requests?**
- Force-close and reopen the Shark app
- Log out and log back in to trigger the Auth0 token exchange
- Check that your phone is actually using the proxy (check mitmproxy's event log)
- Try toggling WiFi off/on on your phone

**Token expires quickly?**
- The refresh token is designed to rotate on each use — this is expected
- The code automatically saves the new token to `config/.shark_token`
- Only the initial capture requires the manual steps above

```yaml
# config/settings.yaml
robot:
  enabled: true
  room_name: "Litter"           # Room name from Shark app
  type: "shark"
  household_id: "YOUR_HOUSEHOLD_ID"
  dsn: "YOUR_DSN"
  dispatch_delay_seconds: 5     # Wait 5s after cooldown
  emergency_stop_on_cat_detected: true
```

### 4. Model Setup

Download or convert a YOLOv8 model:

```bash
# Option A: Convert yourself
pip install ultralytics
yolo export model=yolov8n.pt format=coreml nms=True imgsz=640
mv yolov8n.mlpackage models/

# Option B: Use pre-converted (see models/ directory)
```

### 5. Run the Monitor

```bash
python -m src.main
```

The system will:
1. Start the web dashboard at `http://localhost:8080`
2. Connect to your camera
3. Auto-detect the litter box (green box will appear)
4. Begin monitoring for cats
5. Dispatch the robot when appropriate

---

## 🎛️ Configuration

Complete `config/settings.yaml` reference:

```yaml
# Camera settings
camera:
  rtsp_url: "rtsp://YOUR_CAMERA_IP/live0"
  fps: 15
  resolution: [640, 480]
  use_tcp: true
  reconnect_interval: 5

# Detection settings
detection:
  provider: "coreml_yolo"
  model_path: "models/yolov8n.mlpackage"
  confidence_threshold: 0.5
  inference_interval: 0.2        # 5 FPS detection
  dynamic_zones: true            # Auto-detect litter box
  target_classes: ["cat"]        # What to detect

# State machine timing
timing:
  min_occupancy_seconds: 5       # Must see cat 5s to confirm
  cooldown_seconds: 60           # Wait 60s after exit
  max_session_minutes: 10        # Timeout if cat sleeps there

# Shark robot integration
robot:
  enabled: true
  room_name: "Litter"
  type: "shark"
  household_id: "YOUR_HOUSEHOLD_ID"
  dsn: "YOUR_DSN"
  dispatch_delay_seconds: 5
  emergency_stop_on_cat_detected: true

# Web dashboard
dashboard:
  enabled: true
  host: "0.0.0.0"
  port: 8080

# Event logging
events:
  log_dir: "./logs"
  max_history: 1000

# Visualization
visualization:
  enabled: true
  show_zones: true
```

---

## 📊 Web Dashboard

Access at `http://localhost:8080`:

![Dashboard Preview](https://via.placeholder.com/800x400/2d2d2d/ffffff?text=Live+Dashboard+with+MJPEG+Stream)

**Features:**
- **Live MJPEG stream** with detection overlays
- **Real-time state display** (color-coded)
- **Today's statistics** (visits, duration, last visit)
- **System health** (FPS, latency, uptime)
- **Event log** with auto-scroll

**API Endpoints:**
- `GET /api/stats` — Current system stats
- `GET /api/events` — Recent events
- `GET /video_feed` — MJPEG stream
- `WS /ws` — WebSocket for live updates

---

## 🧪 Model Training (Optional)

Train a custom litter box detection model:

```bash
# 1. Capture training frames
python capture_training_frames.py --rtsp-url rtsp://... --output training_data/

# 2. Annotate with YOLO format
# Place images in training_data/images/ and annotations in training_data/labels/

# 3. Train
python train_yolo.py --data training_data/ --epochs 100

# 4. Export to CoreML
python export_coreml.py --weights runs/detect/best.pt
```

---

## 🖥️ Keyboard Controls

When the video window is open:

| Key | Action |
|-----|--------|
| `q` | Quit the monitor |
| `r` | Reset litter box detection (use if camera moved) |

---

## 🔧 Troubleshooting

### Robot Won't Connect

```bash
# Check token file exists and is valid
cat config/.shark_token

# Verify credentials in settings.yaml
grep -E "household_id|dsn" config/settings.yaml
```

### High Latency

```bash
# Check camera stream latency
ffplay -fflags nobuffer -flags low_delay rtsp://your-camera-url

# Reduce camera resolution in settings.yaml
camera:
  resolution: [480, 360]  # Lower = faster
```

### Litter Box Not Detected

1. Ensure good lighting on the litter box
2. Press `r` to reset detection
3. Check that the box is rectangular (Apple Vision looks for rectangles)
4. Consider using static zones as fallback:

```yaml
detection:
  dynamic_zones: false
  
zones:
  litter_box_1:
    name: "Main Litter Box"
    bbox: [230, 250, 352, 456]  # x1, y1, x2, y2
```

---

## 📁 Project Structure

```
Robot_Vacuum_Litter/
├── src/
│   ├── main.py                 # Entry point
│   ├── camera/
│   │   └── rtsp_client.py      # Low-latency RTSP capture
│   ├── detection/
│   │   ├── coreml_detector.py  # YOLOv8 CoreML inference
│   │   └── litter_box_detector.py  # Dynamic zone detection
│   ├── state/
│   │   └── fsm.py              # Litter box state machine
│   ├── robot/
│   │   ├── interface.py        # Base robot controller
│   │   ├── stub.py             # Stub for testing
│   │   └── shark.py            # 🦈 Shark Matrix integration
│   ├── dashboard/
│   │   └── server.py           # FastAPI web dashboard
│   └── config.py               # Settings management
├── config/
│   ├── settings.yaml           # Main configuration
│   └── .shark_token            # Auth credentials (gitignored)
├── models/                     # CoreML models
└── logs/                       # Event logs
```

---

## 🛡️ Safety & Privacy

- **Local processing only** — Video never leaves your Mac
- **No cloud AI** — All inference runs on Apple Neural Engine
- **Emergency stop** — Robot stops immediately if cat returns
- **Grace periods** — 3-second detection gaps don't trigger false exits
- **Session timeouts** — 10-minute max session prevents stuck states

---

## 🤝 Contributing

This project was built for personal use but contributions are welcome! Areas for improvement:

- [ ] Multi-box support (detect multiple litter boxes)
- [ ] Linux support (non-Apple-Vision fallback)
- [ ] Additional robot brands (Roomba, Roborock, etc.)
- [ ] Mobile app for remote monitoring

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model
- [FastAPI](https://fastapi.tiangolo.com/) for the dashboard backend
- The reverse-engineering community for Shark API documentation

---

<div align="center">

**Made with 🐱☕🤖 on Apple Silicon**

</div>
