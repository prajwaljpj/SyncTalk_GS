#!/usr/bin/env python3
"""
Agent Worker - Handles conversation using Gemini and sends TTS audio to avatar worker.

This worker:
1. Connects to a LiveKit room
2. Creates an AgentSession for conversation (STT, LLM, TTS)
3. Launches an avatar worker subprocess to generate video
4. Sends TTS audio to the avatar via DataStream (data channel)
5. Does NOT publish audio/video itself (avatar does that)

Architecture:
    Agent Worker ──[DataStream]──> Avatar Worker ──[audio+video]──> Client
"""

import os
import sys
import asyncio
import logging
import subprocess
from pathlib import Path

from livekit import agents, api, rtc
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF, RoomOutputOptions
from livekit.plugins import google

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("websockets").setLevel(logging.ERROR)

# --- Configuration from environment ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL", "gemini-live-2.5-flash-preview-native-audio-09-2025"
)

# Avatar configuration (passed to avatar worker via env vars)
SYNCTALK_DATA_PATH = os.environ.get("SYNCTALK_DATA_PATH", "data/May")
SYNCTALK_WORKSPACE = os.environ.get("SYNCTALK_WORKSPACE", "model/trial_may")
SYNCTALK_PORTRAIT = os.environ.get("SYNCTALK_PORTRAIT", "true")
SYNCTALK_TORSO = os.environ.get("SYNCTALK_TORSO", "false")
AVATAR_WIDTH = os.environ.get("AVATAR_WIDTH", "512")
AVATAR_HEIGHT = os.environ.get("AVATAR_HEIGHT", "512")
AVATAR_FPS = os.environ.get("AVATAR_FPS", "25")

# Avatar worker identity
AVATAR_IDENTITY = "avatar_worker"


async def launch_avatar_worker(
    ctx: agents.JobContext, avatar_identity: str
) -> subprocess.Popen:
    """
    Launch the avatar worker as a subprocess.

    Args:
        ctx: JobContext containing room and connection info
        avatar_identity: Identity for the avatar worker

    Returns:
        subprocess.Popen object for the avatar worker process
    """
    # Create a token for the avatar to join the room
    token = (
        api.AccessToken()
        .with_identity(avatar_identity)
        .with_name("Avatar Worker")
        .with_grants(api.VideoGrants(room_join=True, room=ctx.room.name))
        .with_kind("agent")
        .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: ctx.token_claims().identity})
        .to_jwt()
    )

    # Prepare environment variables for avatar worker
    env = os.environ.copy()
    env["LIVEKIT_URL"] = ctx._info.url
    env["LIVEKIT_TOKEN"] = token
    env["SYNCTALK_DATA_PATH"] = SYNCTALK_DATA_PATH
    env["SYNCTALK_WORKSPACE"] = SYNCTALK_WORKSPACE
    env["SYNCTALK_PORTRAIT"] = SYNCTALK_PORTRAIT
    env["SYNCTALK_TORSO"] = SYNCTALK_TORSO
    env["AVATAR_WIDTH"] = AVATAR_WIDTH
    env["AVATAR_HEIGHT"] = AVATAR_HEIGHT
    env["AVATAR_FPS"] = AVATAR_FPS

    # Launch avatar worker subprocess
    avatar_script = Path(__file__).parent / "avatar_worker.py"
    cmd = [sys.executable, str(avatar_script)]

    logger.info(f"Launching avatar worker subprocess...")
    logger.info(f"  Command: {' '.join(cmd)}")
    logger.info(f"  Avatar identity: {avatar_identity}")

    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    logger.info(f"✅ Avatar worker launched (PID: {process.pid})")
    return process


async def entrypoint(ctx: agents.JobContext):
    """
    Main entrypoint for the agent worker.

    This agent:
    1. Connects to a room
    2. Launches an avatar worker subprocess
    3. Listens to user audio via AgentSession
    4. Generates responses using Google's Gemini model
    5. Sends TTS audio to avatar via DataStream
    6. Avatar publishes synchronized audio+video
    """
    logger.info(f"Starting agent in room: {ctx.room.name}")

    # Connect to the room
    await ctx.connect()
    logger.info(f"✅ Connected to room: {ctx.room.name}")

    # Create the Gemini LLM model
    llm_model = google.beta.realtime.RealtimeModel(
        vertexai=True,
        project=GCP_PROJECT_ID,
        location=GCP_REGION,
        model=GEMINI_MODEL,
        voice="Kore",
    )

    # Create the voice agent with instructions
    voice_agent = Agent(
        instructions=(
            "You are a helpful AI assistant with an animated avatar representing customer support for Gnani AI. "
            "Keep your responses conversational and concise. "
            "Speak naturally in an Indian accent as if you're having a face-to-face conversation."
        ),
    )

    # Create AgentSession
    session = AgentSession(
        llm=llm_model,
        resume_false_interruption=False,
    )

    # Launch the avatar worker subprocess
    avatar_process = await launch_avatar_worker(ctx, AVATAR_IDENTITY)

    # Configure DataStreamAudioOutput to send TTS audio to avatar
    # This sends audio via data channel instead of publishing as a track
    logger.info("Configuring DataStream audio output to avatar...")
    logger.info(f"Current audio output type: {type(session.output.audio).__name__}")

    session.output.audio = DataStreamAudioOutput(
        ctx.room,
        destination_identity=AVATAR_IDENTITY,
        # Wait for avatar to publish video before responding (optional)
        wait_remote_track=rtc.TrackKind.KIND_VIDEO,
    )

    logger.info(f"New audio output type: {type(session.output.audio).__name__}")

    # Set up playback finished event handler
    @session.output.audio.on("playback_finished")
    def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
        logger.info(
            "Playback finished",
            extra={
                "playback_position": ev.playback_position,
                "interrupted": ev.interrupted,
            },
        )

    # Start the AgentSession with audio output disabled
    # The avatar worker will publish the audio, not this agent
    logger.info("Starting AgentSession...")
    logger.info(f"  LLM: {type(llm_model).__name__}")
    logger.info(f"  Agent: {type(voice_agent).__name__}")
    logger.info(f"  Audio output: {type(session.output.audio).__name__}")

    # Monitor agent state changes to control avatar behavior
    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        logger.info(f"🔄 Agent state: {event.old_state} → {event.new_state}")

    await session.start(
        agent=voice_agent,
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            audio_enabled=False,  # Avatar publishes audio, not us
            transcription_enabled=True,
        ),
    )
    logger.info("✅ Agent session started")

    # Debug: Check what tracks are published by this agent
    await asyncio.sleep(1.0)  # Wait for any async publishing
    logger.info("📊 Checking published tracks from agent worker:")
    for pub in ctx.room.local_participant.track_publications.values():
        logger.info(f"  - {pub.kind.name}: {pub.name} (source: {pub.source.name})")

    logger.info("✅ Agent is ready - listening for user input")

    # Keep the agent running until the room disconnects
    try:

        async def wait_until_disconnected():
            while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                await asyncio.sleep(1)

        await wait_until_disconnected()
        logger.info("Room disconnected, shutting down...")

    except Exception as e:
        logger.error(f"Agent session error: {e}")

    finally:
        # Cleanup: Terminate avatar worker subprocess
        logger.info("Shutting down agent...")

        if avatar_process and avatar_process.poll() is None:
            logger.info(f"Terminating avatar worker (PID: {avatar_process.pid})")
            avatar_process.terminate()
            try:
                avatar_process.wait(timeout=5)
                logger.info("✅ Avatar worker terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Avatar worker did not terminate, killing...")
                avatar_process.kill()
                avatar_process.wait()
                logger.info("✅ Avatar worker killed")

        logger.info("✅ Agent shutdown complete")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
