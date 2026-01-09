#!/bin/bash
# Start LiveKit + Gemini + Custom Talking Head Agent
#
# This script configures environment variables needed by main_agent.py
# and custom_avatar_worker.py, then starts the main agent process.

# --- CHECK FOR REQUIRED CREDENTIALS ---
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ ERROR: GOOGLE_APPLICATION_CREDENTIALS must be set for Vertex AI."
    echo ""
    echo "Please set:"
    echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/service-account.json\""
    echo "  export VERTEX_PROJECT_ID=\"your-gcp-project-id\""
    echo "  export VERTEX_LOCATION=\"us-central1\""
    echo ""
    echo "Then run: ./start_main_agent.sh"
    exit 1
fi

# Display which auth method is being used
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Using Vertex AI authentication"
    echo "Credentials: $GOOGLE_APPLICATION_CREDENTIALS"
    if [ -z "$VERTEX_PROJECT_ID" ]; then
        echo "⚠️  WARNING: VERTEX_PROJECT_ID not set, agent may fail"
    fi
fi

# --- 1. LIVEKIT CONFIGURATION ---
# Default to local development server credentials if not set
export LIVEKIT_URL=${LIVEKIT_URL:-ws://localhost:7880}
export LIVEKIT_API_KEY=${LIVEKIT_API_KEY:-devkey}
export LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET:-devsecret}

# --- 2. VERTEX AI CONFIGURATION ---
# Map the script's variables to the names used in main_agent.py
export GCP_PROJECT_ID=${VERTEX_PROJECT_ID:-$GCP_PROJECT_ID}
export GCP_REGION=${VERTEX_LOCATION:-us-central1}

# Set the specific Gemini model you are using
export GEMINI_MODEL=${GEMINI_MODEL:-gemini-live-2.5-flash-preview-native-audio-09-2025}

# --- 3. SYNCTALK CONFIGURATION ---
# Configure SyncTalk model paths
export SYNCTALK_DATA_PATH=${SYNCTALK_DATA_PATH:-data/May}
export SYNCTALK_WORKSPACE=${SYNCTALK_WORKSPACE:-model/trial_may}
export SYNCTALK_PORTRAIT=${SYNCTALK_PORTRAIT:-true}
export SYNCTALK_TORSO=${SYNCTALK_TORSO:-false}

# Avatar output configuration
export AVATAR_WIDTH=${AVATAR_WIDTH:-512}
export AVATAR_HEIGHT=${AVATAR_HEIGHT:-512}
export AVATAR_FPS=${AVATAR_FPS:-25}

echo "======================================================"
echo "Starting LiveKit + Vertex AI + SyncTalk Avatar Agent"
echo "======================================================"
echo "Architecture: Two-Worker System"
echo "  1. Agent Worker: Conversation (STT, LLM, TTS)"
echo "  2. Avatar Worker: Video Generation (SyncTalk)"
echo "  Audio sent via DataStream (agent → avatar)"
echo "======================================================"
echo "LiveKit Server:  $LIVEKIT_URL"
echo ""
echo "SyncTalk Config:"
echo "  Data Path:     $SYNCTALK_DATA_PATH"
echo "  Workspace:     $SYNCTALK_WORKSPACE"
echo "  Portrait:      $SYNCTALK_PORTRAIT"
echo "  Torso:         $SYNCTALK_TORSO"
echo "  Resolution:    ${AVATAR_WIDTH}x${AVATAR_HEIGHT} @ ${AVATAR_FPS}fps"
echo ""
echo "Vertex AI Config:"
echo "  Model:         $GEMINI_MODEL"
echo "  Project ID:    $GCP_PROJECT_ID"
echo "  Region:        $GCP_REGION"
echo "======================================================"
echo ""

# --- 4. CHECK DEPENDENCIES ---
# Note: Assumes you're running from UV environment (.venv-livekit)
# Check for scipy
if ! python -c "import scipy.signal" 2>/dev/null; then
    echo "⚠️  Installing scipy for audio resampling..."
    pip install scipy
fi

# Check for google-genai
# Note: The 'google' LiveKit plugin relies on google-genai (Python 3.9+) or google-cloud-aiplatform (Python 3.8+).
# Using google-genai for Python 3.10+.
if ! python -c "import google.genai" 2>/dev/null; then
    echo "⚠️  Installing google-genai..."
    pip install google-genai
fi

# --- 5. RUN THE AGENT WORKER ---
echo "Starting agent worker..."
echo "(Avatar worker will be launched automatically as subprocess)"
echo ""
# Execute the agent worker (which launches avatar worker)
python livekit/server/agent_worker.py dev
