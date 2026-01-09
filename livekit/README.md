# LiveKit Integration for SyncTalk

Real-time conversational avatar using SyncTalk's NeRF-based video generation with LiveKit infrastructure.

## 📁 Directory Structure

```
livekit/
├── server/                    # Backend server components
│   ├── agent_worker.py       # Conversation handler (STT/LLM/TTS)
│   ├── avatar_worker.py      # Video generation handler
│   ├── synctalk_video_generator.py  # SyncTalk video generator
│   └── ditto_video_generator.py     # Old Ditto generator (reference)
│
├── client/                    # Frontend test client
│   ├── simple_client.html    # Web-based test client
│   └── token_server.py       # JWT token server for testing
│
├── docs/                      # Documentation
│   ├── LIVEKIT_INTEGRATION_GUIDE.md   # Detailed integration guide
│   └── LIVEKIT_SETUP_COMPLETE.md      # Quick start guide
│
├── livekit_server.sh         # Startup script
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Trained SyncTalk model** (in `../data/` and `../model/`)
2. **Google Cloud credentials** for Gemini API
3. **LiveKit server** running (Docker or local)

### Start the Server

```bash
# From project root
cd /path/to/SyncTalk

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export VERTEX_PROJECT_ID="your-gcp-project-id"
export VERTEX_LOCATION="us-central1"

# Optional: Configure SyncTalk model
export SYNCTALK_DATA_PATH="data/May"
export SYNCTALK_WORKSPACE="model/trial_may"
export SYNCTALK_PORTRAIT="true"

# Start the agent
./livekit/livekit_server.sh
```

### Test with Web Client

```bash
# Terminal 1: Start token server
cd livekit/client
python token_server.py

# Terminal 2: Open browser
open http://localhost:8080/simple_client.html
```

## 📖 Documentation

- **[Quick Start Guide](docs/LIVEKIT_SETUP_COMPLETE.md)** - Setup instructions, troubleshooting
- **[Integration Guide](docs/LIVEKIT_INTEGRATION_GUIDE.md)** - Architecture details, implementation

## 🏗️ Architecture

```
Agent Worker ──[DataStream]──> Avatar Worker ──[audio+video]──> Client
     ↓                                ↓
  Gemini API                    SyncTalk NeRF
  (STT/LLM/TTS)                 (Video Generation)
```

**Two-worker system**:
1. **Agent Worker**: Handles conversation using Gemini (STT, LLM, TTS)
2. **Avatar Worker**: Generates video using SyncTalk, publishes to room

Communication via LiveKit DataStream (data channel).

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNCTALK_DATA_PATH` | `data/May` | Training data directory |
| `SYNCTALK_WORKSPACE` | `model/trial_may` | Model checkpoint directory |
| `SYNCTALK_PORTRAIT` | `true` | Enable portrait mode |
| `SYNCTALK_TORSO` | `false` | Enable torso mode |
| `AVATAR_WIDTH` | `512` | Output video width |
| `AVATAR_HEIGHT` | `512` | Output video height |
| `AVATAR_FPS` | `25` | Output FPS (must be 25) |

### LiveKit Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVEKIT_URL` | `ws://localhost:7880` | LiveKit server URL |
| `LIVEKIT_API_KEY` | `devkey` | API key (dev mode) |
| `LIVEKIT_API_SECRET` | `devsecret` | API secret (dev mode) |

## 🎯 Features

- ✅ Real-time lip-sync using SyncTalk NeRF
- ✅ ~415-500ms end-to-end latency
- ✅ Photorealistic facial expressions
- ✅ Idle animation when not speaking
- ✅ Portrait mode for face compositing
- ✅ Gemini-powered conversation (STT/LLM/TTS)

## 🔧 Development

### Running Individual Components

```bash
# Agent worker only (for testing)
uv run python livekit/server/agent_worker.py dev

# Avatar worker (launched automatically by agent)
# Requires LIVEKIT_URL and LIVEKIT_TOKEN environment variables
```

### Debugging

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
./livekit/livekit_server.sh
```

### Testing Changes

1. Modify `synctalk_video_generator.py`
2. Restart server: `Ctrl+C` then `./livekit/livekit_server.sh`
3. Test in browser client

## 📊 Performance

| Metric | Value |
|--------|-------|
| Latency (total) | ~415-500ms |
| GPU Memory | ~4-6GB VRAM |
| CPU Usage | ~10-20% |
| Network Bandwidth | ~2-4 Mbps |
| Video Quality | Photorealistic |

## 🐛 Troubleshooting

### No video appears
- Check logs for "Connected to room"
- Verify SyncTalk model loaded
- Check GPU availability

### Audio/video out of sync
- Ensure FPS=25
- Check queue sizes in logs
- Verify chunk size (1280 samples)

### High latency
- Check GPU utilization (should be >80%)
- Verify network RTT
- Consider disabling portrait mode

See [LIVEKIT_SETUP_COMPLETE.md](docs/LIVEKIT_SETUP_COMPLETE.md) for more troubleshooting tips.

## 📝 License

Same as parent project (SyncTalk).

## 🤝 Contributing

When contributing to the LiveKit integration:
1. Keep server components in `server/`
2. Keep client components in `client/`
3. Update documentation in `docs/`
4. Test with the web client before submitting

## 🔗 Related Files

- `../streaming_inference.py` - Core SyncTalk streaming inference
- `../example_streaming.py` - Standalone example
- `../main.py` - Original batch inference

---

**Need help?** See the [documentation](docs/) or open an issue.
