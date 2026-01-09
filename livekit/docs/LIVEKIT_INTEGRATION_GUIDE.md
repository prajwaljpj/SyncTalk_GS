# LiveKit Integration Guide for SyncTalk

## Architecture Overview

Your LiveKit setup uses a **two-worker architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                         LiveKit Room                             │
│                                                                   │
│  ┌──────────────────┐                    ┌──────────────────┐   │
│  │  Agent Worker    │                    │  Avatar Worker   │   │
│  │                  │                    │                  │   │
│  │  1. STT (Gemini) │                    │  4. Generate     │   │
│  │  2. LLM (Gemini) │  ──[DataStream]──> │     Video        │   │
│  │  3. TTS (Gemini) │     (audio)        │     (SyncTalk)   │   │
│  │                  │                    │                  │   │
│  └──────────────────┘                    │  5. Publish      │   │
│                                          │     audio+video  │   │
│                                          │                  │   │
│                                          └─────────┬────────┘   │
│                                                    │             │
│                               ┌────────────────────┘             │
│                               ▼                                  │
│                        ┌──────────────┐                          │
│                        │    Client    │                          │
│                        │  (Browser)   │                          │
│                        └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

### Current Setup (Ditto)

1. **Agent Worker** (`agent_worker.py`):
   - Listens to user audio (STT)
   - Processes with Gemini LLM
   - Generates TTS audio (Gemini voice)
   - Sends audio via DataStream to avatar worker
   - Does NOT publish audio/video itself

2. **Avatar Worker** (`avatar_worker.py`):
   - Receives TTS audio from agent via DataStream
   - Passes audio to video generator (Ditto)
   - Publishes synchronized audio+video to room
   - Uses `AvatarRunner` for synchronization

3. **Video Generator** (`ditto_video_generator.py`):
   - Implements `VideoGenerator` interface (LiveKit standard)
   - Receives audio chunks
   - Generates video frames
   - Pairs audio+video for output

### Key Components

| Component | Purpose | Keep/Replace |
|-----------|---------|--------------|
| `livekit_server.sh` | Start script | ✅ Keep (update env vars) |
| `agent_worker.py` | Conversation handling | ✅ Keep (no changes needed) |
| `avatar_worker.py` | Video publishing | ✅ Keep (minimal changes) |
| `ditto_video_generator.py` | Video generation | ❌ Replace with SyncTalk |
| `livekit_client/` | Test client | ✅ Keep (no changes) |

## Integration Steps

### Step 1: Create SyncTalk Video Generator

You need to create `synctalk_video_generator.py` to replace `ditto_video_generator.py`.

**Key Requirements:**

1. **Implement `VideoGenerator` interface**:
   ```python
   class VideoGenerator(ABC):
       async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None
       def clear_buffer(self) -> None
       def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]
   ```

2. **Use SyncTalk's StreamingInference**:
   ```python
   from streaming_inference import StreamingInference

   self.inference = StreamingInference(
       data_path="data/May",
       workspace="model/trial_may",
       portrait=True,
       torso=False
   )
   self.inference.reset_streaming_buffers()
   ```

3. **Process audio chunks**:
   ```python
   # Receive TTS audio from agent
   audio_samples = convert_to_16khz_mono(audio_frame)

   # Process through SyncTalk
   frames = self.inference.process_audio_chunk(audio_samples)

   # Yield video+audio pairs
   for video_frame in frames:
       yield video_frame
       yield corresponding_audio
   ```

### Step 2: Adapt Chunking Parameters

**Ditto Chunking:**
- Chunk size: 6480 samples (405ms)
- Stride: 3200 samples (200ms)
- Context: (3, 5, 2) past/current/future frames

**SyncTalk Chunking:**
- Chunk size: 1280 samples (80ms) **recommended**
- Stride: Same as chunk (no overlap)
- Context: Built into model (bi-directional attention)

**Key Difference:**
- Ditto: Manual chunking with overlap
- SyncTalk: Automatic buffering in `process_audio_chunk()`

### Step 3: Audio Format Matching

Both systems use the same format (perfect!):

| Parameter | Ditto | SyncTalk | Match |
|-----------|-------|----------|-------|
| Sample Rate | 16kHz | 16kHz | ✅ |
| Channels | Mono | Mono | ✅ |
| Format | float32 | float32 | ✅ |
| FPS | 25 | 25 | ✅ |

**No resampling needed!**

### Step 4: Handle AudioSegmentEnd

When agent finishes speaking, it sends `AudioSegmentEnd`:

```python
if isinstance(frame, AudioSegmentEnd):
    # Flush any remaining buffered frames
    # Return to idle animation
    yield AudioSegmentEnd()
```

### Step 5: Idle Animation

When no TTS audio is playing, generate idle frames:

```python
async def _generate_idle_frame(self):
    # Generate neutral expression with silence
    silence = np.zeros(1280, dtype=np.float32)
    frames = self.inference.process_audio_chunk(silence)

    if frames:
        return frames[0], silence_audio_frame
    return None, None
```

## Detailed Implementation

### synctalk_video_generator.py Structure

```python
class SyncTalkVideoGenerator(VideoGenerator):
    def __init__(self, data_path, workspace, portrait=True):
        # Initialize SyncTalk
        self.inference = StreamingInference(...)

        # Queues (same pattern as Ditto)
        self._audio_input_queue = asyncio.Queue()
        self._video_queue = asyncio.Queue()
        self._audio_output_queue = asyncio.Queue()

        # Background processor
        self._processor_task = None

    async def push_audio(self, frame):
        """Called by AvatarRunner when agent sends TTS"""
        await self._audio_input_queue.put(frame)

    async def _process_audio(self):
        """Background task: Process TTS audio through SyncTalk"""
        while self._running:
            frame = await self._audio_input_queue.get()

            if isinstance(frame, AudioSegmentEnd):
                await self._audio_output_queue.put(AudioSegmentEnd())
                continue

            # Convert to numpy
            audio_samples = self._resample_to_16k(frame)

            # Process through SyncTalk
            video_frames = self.inference.process_audio_chunk(audio_samples)

            # Queue frames
            for video_frame in video_frames:
                # Convert numpy to LiveKit VideoFrame
                livekit_frame = self._numpy_to_videoframe(video_frame)
                await self._video_queue.put(livekit_frame)

                # Pair with corresponding audio
                audio_chunk = self._extract_audio_chunk(audio_samples)
                await self._audio_output_queue.put(audio_chunk)

    async def _stream_impl(self):
        """Main generator: Yields frames to AvatarRunner"""
        while self._running:
            try:
                # Try to get TTS audio
                audio = await asyncio.wait_for(
                    self._audio_output_queue.get(),
                    timeout=0.05
                )
            except asyncio.TimeoutError:
                # No TTS - generate idle frame
                video, audio = await self._generate_idle_frame()
                if video and audio:
                    yield video
                    yield audio
                continue

            # Handle segment end
            if isinstance(audio, AudioSegmentEnd):
                yield AudioSegmentEnd()
                continue

            # Get corresponding video
            video = await self._video_queue.get()

            # Yield pair
            yield video
            yield audio
```

### Key Differences from Ditto

| Aspect | Ditto | SyncTalk |
|--------|-------|----------|
| **Chunking** | Manual (6480 samples) | Automatic (1280 samples) |
| **Buffering** | Custom buffer logic | Built-in `process_audio_chunk()` |
| **Context** | (3,5,2) manual window | Bi-directional attention (automatic) |
| **Processing** | `sdk.run_chunk()` blocking | `process_audio_chunk()` sync |
| **Callbacks** | Async callback from SDK | Direct return value |
| **Threading** | SDK runs in worker threads | Runs in asyncio event loop |

### Simplified Flow

**Ditto** (Complex):
1. Buffer audio manually
2. Check if buffer >= 6480 samples
3. Extract chunk with context
4. Call `sdk.run_chunk()` in executor
5. Wait for callback from SDK thread
6. Pair video with audio in callback
7. Queue to output

**SyncTalk** (Simple):
1. Call `process_audio_chunk(audio_samples)`
2. Get list of video frames immediately
3. Pair each frame with corresponding audio
4. Queue to output

**SyncTalk is much simpler!** No manual buffering, no callbacks, no threading!

## Configuration Changes

### livekit_server.sh

Replace Ditto env vars with SyncTalk:

```bash
# OLD (Ditto)
export DATA_ROOT=${DITTO_DATA_ROOT:-checkpoints/ditto_trt_custom2/}
export CFG_PKL=${DITTO_CFG_PKL:-checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl}
export SOURCE_PATH=${DITTO_SOURCE:-avatars/source.jpg}

# NEW (SyncTalk)
export SYNCTALK_DATA_PATH=${SYNCTALK_DATA_PATH:-data/May}
export SYNCTALK_WORKSPACE=${SYNCTALK_WORKSPACE:-model/trial_may}
export SYNCTALK_PORTRAIT=${SYNCTALK_PORTRAIT:-true}
export SYNCTALK_TORSO=${SYNCTALK_TORSO:-false}
```

### agent_worker.py

No changes needed! Just update env var names:

```python
# OLD
DATA_ROOT = os.environ.get("DATA_ROOT", "./checkpoints/...")
CFG_PKL = os.environ.get("CFG_PKL", "./checkpoints/...")
SOURCE_PATH = os.environ.get("SOURCE_PATH", "./assets/...")

# NEW
SYNCTALK_DATA_PATH = os.environ.get("SYNCTALK_DATA_PATH", "data/May")
SYNCTALK_WORKSPACE = os.environ.get("SYNCTALK_WORKSPACE", "model/trial_may")
SYNCTALK_PORTRAIT = os.environ.get("SYNCTALK_PORTRAIT", "true") == "true"
SYNCTALK_TORSO = os.environ.get("SYNCTALK_TORSO", "false") == "true"
```

### avatar_worker.py

Change import only:

```python
# OLD
from ditto_video_generator import DittoVideoGenerator

# NEW
from synctalk_video_generator import SyncTalkVideoGenerator

# OLD
video_gen = DittoVideoGenerator(
    cfg_pkl=CFG_PKL,
    data_root=DATA_ROOT,
    source_path=SOURCE_PATH,
    ...
)

# NEW
video_gen = SyncTalkVideoGenerator(
    data_path=SYNCTALK_DATA_PATH,
    workspace=SYNCTALK_WORKSPACE,
    portrait=SYNCTALK_PORTRAIT,
    video_width=AVATAR_WIDTH,
    video_height=AVATAR_HEIGHT,
    video_fps=AVATAR_FPS,
)
```

## Audio-Video Synchronization

### Timing Calculations

At 25 FPS:
- 1 frame = 40ms = 640 samples @ 16kHz
- Chunk size = 1280 samples = 80ms = 2 frames

**Processing Flow:**
```
TTS Audio Chunk (1280 samples = 80ms)
    ↓
SyncTalk.process_audio_chunk()
    ↓
Returns 0-2 video frames (depends on buffering)
    ↓
Pair each frame with 640 samples (40ms)
    ↓
Yield to LiveKit
```

### Latency Analysis

**Total Latency** (from agent speaking to client seeing video):

| Stage | Time | Notes |
|-------|------|-------|
| Agent TTS generation | ~50-100ms | Gemini voice |
| DataStream transfer | ~10-20ms | In-memory |
| SyncTalk buffering | ~320ms | Bi-directional attention |
| SyncTalk rendering | ~20-30ms | Per frame |
| LiveKit encoding | ~10-20ms | H.264 |
| Network (local) | ~5-10ms | WebRTC |
| **Total** | **~415-500ms** | Half-second delay |

**This is excellent for conversational AI!** (Under 1 second is considered real-time)

### Ditto vs SyncTalk Latency

| System | Latency | Quality | Speed |
|--------|---------|---------|-------|
| Ditto | ~300-400ms | Good | Fast (TensorRT) |
| SyncTalk | ~415-500ms | Excellent | Moderate (NeRF) |

**Trade-off:** SyncTalk adds ~100ms latency but produces much better quality.

## Testing

### 1. Start LiveKit Server

```bash
# Terminal 1: Start LiveKit server (if not running)
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
    livekit/livekit-server --dev
```

### 2. Start SyncTalk Agent

```bash
# Terminal 2: Export credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export VERTEX_PROJECT_ID="your-gcp-project-id"
export VERTEX_LOCATION="us-central1"

# Set SyncTalk config
export SYNCTALK_DATA_PATH="data/May"
export SYNCTALK_WORKSPACE="model/trial_may"
export SYNCTALK_PORTRAIT="true"

# Run server
./livekit_server.sh
```

### 3. Open Test Client

```bash
# Terminal 3: Start token server
cd livekit_client
python token_server.py

# Open browser
open http://localhost:8080/simple_client.html
```

### 4. Expected Behavior

1. Client connects to room
2. Agent worker starts
3. Avatar worker spawns as subprocess
4. Client sees avatar video (idle animation)
5. User speaks → STT → LLM → TTS
6. Avatar lip-syncs to TTS audio
7. After speaking, returns to idle

## Troubleshooting

### Issue: No video appears

**Check:**
1. Avatar worker connected? (Check logs for "Connected to room")
2. Video track published? (Check "track_publications" log)
3. SyncTalk model loaded? (Check for "Model loaded" message)

### Issue: Video/audio out of sync

**Check:**
1. Audio chunk size (should be 1280 samples)
2. Video frame count matches audio chunks
3. No queue overflow (check queue sizes in logs)

### Issue: High latency (>1 second)

**Check:**
1. GPU utilization (should be >80%)
2. Network RTT (use `ping` to LiveKit server)
3. Portrait mode enabled? (slower but better quality)

### Issue: Choppy video

**Check:**
1. FPS = 25 (both sides)
2. No frame drops (check "skipped frames" in logs)
3. Sufficient bandwidth

## Performance Optimization

### GPU Memory

SyncTalk uses ~4-6GB VRAM:
- NeRF model: ~2GB
- Audio encoder: ~500MB
- Rendering buffers: ~1-2GB
- PyTorch overhead: ~1GB

**For multiple concurrent sessions:** Use batching or model sharing

### CPU Usage

- Agent worker: ~10-20% (Gemini API)
- Avatar worker: ~5-10% (queuing only)
- SyncTalk: GPU-bound (CPU <5%)

### Network Bandwidth

At 1280x720 @ 25 FPS:
- Uncompressed: ~66 Mbps
- H.264 (medium): ~2-4 Mbps
- H.264 (low): ~1-2 Mbps

## Next Steps

1. **Create `synctalk_video_generator.py`** (see template above)
2. **Test with simple audio** (ensure frames generate correctly)
3. **Test with LiveKit client** (end-to-end integration)
4. **Optimize for production** (error handling, monitoring)
5. **Deploy** (Docker, GPU server, etc.)

## Summary

**What to do:**
1. ✅ Keep: `agent_worker.py`, `avatar_worker.py`, `livekit_server.sh`
2. ❌ Replace: `ditto_video_generator.py` → `synctalk_video_generator.py`
3. 🔧 Update: Environment variables (Ditto → SyncTalk)

**Key advantages of SyncTalk integration:**
- ✅ Better lip-sync quality (NeRF vs. simple warping)
- ✅ More realistic facial expressions
- ✅ Portrait mode for photorealistic results
- ✅ Same audio format (16kHz mono)
- ✅ Same FPS (25)
- ✅ Simpler chunking logic (no manual buffering)

**Ready to implement?** Let me know if you want me to create the full `synctalk_video_generator.py` file!
