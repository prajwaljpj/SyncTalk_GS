#!/usr/bin/env python3
"""
SyncTalk Video Generator - LiveKit Integration

Generates lip-synced video from audio using SyncTalk's NeRF-based renderer.
This is a significantly simpler implementation than Ditto because SyncTalk
handles buffering and context automatically in process_audio_chunk().

Key advantages over Ditto:
- No manual buffering (SyncTalk handles it)
- No callbacks (direct return values)
- No threading (runs in asyncio)
- Simpler chunking (1280 samples vs 6480)
"""

import os
import sys
import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

# Add project root to path for streaming_inference import
# Path: livekit/server/synctalk_video_generator.py -> project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from streaming_inference import StreamingInference

logger = logging.getLogger(__name__)


class SyncTalkVideoGenerator(VideoGenerator):
    """
    Video generator using SyncTalk's NeRF-based streaming inference.

    Much simpler than Ditto because:
    - SyncTalk handles audio buffering automatically
    - No manual overlap/stride calculations
    - Direct function calls (no SDK threads/callbacks)
    - Smaller chunks (1280 samples = 80ms)
    """

    def __init__(
        self,
        data_path: str,
        workspace: str,
        *,
        portrait: bool = True,
        torso: bool = False,
        video_width: int = 512,
        video_height: int = 512,
        video_fps: int = 25,
    ):
        """
        Initialize SyncTalk video generator.

        Args:
            data_path: Path to training data directory (e.g., 'data/May')
            workspace: Path to trained model workspace (e.g., 'model/trial_may')
            portrait: Enable portrait mode (composites face onto original image)
            torso: Enable torso mode (includes torso in rendering)
            video_width: Output video width
            video_height: Output video height
            video_fps: Output video FPS (must be 25!)
        """
        if video_fps != 25:
            logger.warning(
                f"⚠️ SyncTalk requires FPS=25, got {video_fps}. Forcing FPS=25."
            )
            video_fps = 25

        self._video_width = video_width
        self._video_height = video_height
        self._video_fps = video_fps

        # === AUDIOWAVE PATTERN: Single input queue ===
        self._audio_input_queue: asyncio.Queue[
            rtc.AudioFrame | AudioSegmentEnd
        ] = asyncio.Queue()

        # === OUTPUT QUEUES: For pairing video+audio ===
        self._video_queue: asyncio.Queue[rtc.VideoFrame] = asyncio.Queue()
        self._audio_output_queue: asyncio.Queue[
            rtc.AudioFrame | AudioSegmentEnd
        ] = asyncio.Queue()

        # === SYNCTALK CHUNKING ===
        # SyncTalk recommended chunk size: 1280 samples = 80ms @ 16kHz
        # This generates 2 frames (80ms / 40ms per frame = 2 frames)
        self.chunk_size = 1280  # Much simpler than Ditto's 6480!

        # Each video frame = 640 samples (40ms @ 16kHz @ 25 FPS)
        self.samples_per_frame = 640

        # === INITIALIZE SYNCTALK ===
        logger.info("Initializing SyncTalk StreamingInference...")
        logger.info(f"  Data path: {data_path}")
        logger.info(f"  Workspace: {workspace}")
        logger.info(f"  Portrait: {portrait}")
        logger.info(f"  Resolution: {video_width}x{video_height} @ {video_fps}fps")

        self.inference = StreamingInference(
            data_path=data_path,
            workspace=workspace,
            portrait=portrait,
            torso=torso,
            W=video_width,
            H=video_height,
            fps=video_fps,
        )
        self.inference.reset_streaming_buffers()

        # Background task
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"✅ SyncTalk initialized: {video_width}x{video_height} @ {video_fps}fps"
        )

    # ============================================================================
    # VideoGenerator Interface (Required by LiveKit)
    # ============================================================================

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """
        Push audio from agent to generator.

        Called by AvatarRunner when agent sends TTS audio.
        """
        if isinstance(frame, AudioSegmentEnd):
            logger.info("📨 push_audio: Received AudioSegmentEnd from agent")
        else:
            logger.info(
                f"📨 push_audio: Received TTS audio frame: {frame.samples_per_channel} samples @ {frame.sample_rate}Hz"
            )
        await self._audio_input_queue.put(frame)

    def clear_buffer(self) -> None:
        """
        Clear all buffers on interruption.

        Called by AvatarRunner when user interrupts.
        """
        # Drain all queues
        while not self._audio_input_queue.empty():
            try:
                self._audio_input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._video_queue.empty():
            try:
                self._video_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._audio_output_queue.empty():
            try:
                self._audio_output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Reset SyncTalk buffers (much simpler than Ditto!)
        self.inference.reset_streaming_buffers()

        logger.info("🔄 Buffers cleared")

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        """Return async iterator for frame generation."""
        return self._stream_impl()

    # ============================================================================
    # Background Audio Processing
    # ============================================================================

    async def start(self):
        """Start background audio processor."""
        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self._running = True
        self._processor_task = asyncio.create_task(self._process_audio())
        logger.info("🚀 Background audio processor started")

    async def _process_audio(self):
        """
        Background task: Process incoming audio through SyncTalk.

        Flow (much simpler than Ditto!):
        1. Get audio from input queue
        2. Resample to 16kHz mono
        3. Call process_audio_chunk() - returns list of frames immediately!
        4. Pair each video frame with corresponding audio
        5. Queue both for output

        No manual buffering, no callbacks, no threading!
        """
        while self._running:
            try:
                # Wait for TTS audio from agent
                try:
                    frame = await asyncio.wait_for(
                        self._audio_input_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # No TTS audio - idle generation handled separately
                    await asyncio.sleep(0.001)
                    continue

                # Handle AudioSegmentEnd
                if isinstance(frame, AudioSegmentEnd):
                    logger.info("📨 AudioSegmentEnd received - end of TTS")
                    await self._audio_output_queue.put(AudioSegmentEnd())
                    continue

                # Process AudioFrame (TTS)
                if isinstance(frame, rtc.AudioFrame):
                    # Resample to 16kHz mono
                    audio_samples = self._resample_to_16k(frame)
                    logger.info(
                        f"🎵 TTS AUDIO: Received {len(audio_samples)} samples"
                    )

                    # Process through SyncTalk (returns frames immediately!)
                    video_frames = await self._loop.run_in_executor(
                        None, self.inference.process_audio_chunk, audio_samples
                    )

                    logger.info(
                        f"🎬 SyncTalk generated {len(video_frames)} video frames"
                    )

                    # Pair each video frame with corresponding audio
                    for i, video_np in enumerate(video_frames):
                        # Extract audio chunk for this frame (640 samples = 40ms)
                        audio_start = i * self.samples_per_frame
                        audio_end = audio_start + self.samples_per_frame

                        if audio_end <= len(audio_samples):
                            audio_chunk = audio_samples[audio_start:audio_end]
                        else:
                            # Pad with silence if needed
                            audio_chunk = np.zeros(
                                self.samples_per_frame, dtype=np.float32
                            )
                            if audio_start < len(audio_samples):
                                valid_len = len(audio_samples) - audio_start
                                audio_chunk[:valid_len] = audio_samples[audio_start:]

                        # Convert video frame to LiveKit format
                        video_frame = self._numpy_to_videoframe(video_np)
                        audio_frame = self._numpy_to_audioframe(audio_chunk)

                        # Queue both
                        await self._video_queue.put(video_frame)
                        await self._audio_output_queue.put(audio_frame)

                    logger.info(
                        f"📊 QUEUES: video={self._video_queue.qsize()}, "
                        f"audio_out={self._audio_output_queue.qsize()}"
                    )

            except Exception as e:
                logger.error(f"Audio processor error: {e}", exc_info=True)

    async def _generate_idle_frame(
        self,
    ) -> tuple[Optional[rtc.VideoFrame], Optional[rtc.AudioFrame]]:
        """
        Generate a single idle frame (neutral expression with silence).

        Called by main loop when no TTS audio is available.
        """
        # Generate silence (1 frame = 640 samples)
        silence = np.zeros(self.samples_per_frame, dtype=np.float32)

        # Process through SyncTalk
        video_frames = await self._loop.run_in_executor(
            None, self.inference.process_audio_chunk, silence
        )

        if video_frames:
            # Convert first frame
            video_frame = self._numpy_to_videoframe(video_frames[0])
            audio_frame = self._numpy_to_audioframe(silence)
            return video_frame, audio_frame
        else:
            # Not enough context yet
            return None, None

    # ============================================================================
    # Main Generator (yields frames to AvatarRunner)
    # ============================================================================

    async def _stream_impl(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        """
        Main generator following AudioWave pattern.

        Yields video+audio pairs to AvatarRunner.
        """
        frame_count = 0
        idle_mode = True

        while self._running:
            try:
                # Try to get TTS audio
                try:
                    audio = await asyncio.wait_for(
                        self._audio_output_queue.get(), timeout=0.05
                    )
                    idle_mode = False
                except asyncio.TimeoutError:
                    # No TTS - generate idle frame
                    if not idle_mode:
                        idle_mode = True
                        logger.info("💤 Entering IDLE mode")

                    video, audio = await self._generate_idle_frame()
                    if video and audio:
                        yield video
                        yield audio
                        frame_count += 1
                    else:
                        # Not enough context yet, just wait
                        await asyncio.sleep(0.001)
                    continue

                # Handle AudioSegmentEnd
                if isinstance(audio, AudioSegmentEnd):
                    logger.info("📤 Yielding AudioSegmentEnd")
                    yield AudioSegmentEnd()
                    idle_mode = True
                    continue

                # Got TTS audio - get corresponding video
                if not idle_mode:
                    logger.info("🎤 Entering TTS mode")
                    idle_mode = False

                try:
                    video = await asyncio.wait_for(self._video_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Video not ready for audio - skipping")
                    continue

                # Yield synchronized pair
                yield video
                yield audio

                frame_count += 1
                if frame_count % 25 == 0:
                    logger.info(f"📊 Yielded {frame_count} frame pairs")

            except Exception as e:
                if not self._running:
                    logger.info("Stream generator stopped due to shutdown")
                    break
                logger.error(f"Error in stream generator: {e}", exc_info=True)

        logger.info("Stream generator exited gracefully")

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _resample_to_16k(self, frame: rtc.AudioFrame) -> np.ndarray:
        """
        Resample audio frame to 16kHz mono.

        Returns:
            Float32 array in range [-1, 1]
        """
        # Convert to float32
        audio_data = (
            np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # Handle multi-channel (convert to mono)
        if frame.num_channels > 1:
            audio_data = audio_data.reshape(-1, frame.num_channels).mean(axis=1)

        # Resample if needed
        if frame.sample_rate != 16000:
            from scipy import signal

            # Calculate resampling ratio
            num_samples = int(len(audio_data) * 16000 / frame.sample_rate)
            audio_data = signal.resample(audio_data, num_samples)

        return audio_data.astype(np.float32)

    def _numpy_to_videoframe(self, frame_np: np.ndarray) -> rtc.VideoFrame:
        """
        Convert numpy array to LiveKit VideoFrame.

        Args:
            frame_np: [H, W, 3] numpy array (uint8, RGB)

        Returns:
            rtc.VideoFrame
        """
        # Ensure correct shape and type
        assert frame_np.dtype == np.uint8, f"Expected uint8, got {frame_np.dtype}"
        assert len(frame_np.shape) == 3, f"Expected [H,W,3], got {frame_np.shape}"
        assert frame_np.shape[2] == 3, f"Expected 3 channels, got {frame_np.shape[2]}"

        # Create LiveKit VideoFrame (expects RGB)
        return rtc.VideoFrame(
            width=frame_np.shape[1],
            height=frame_np.shape[0],
            type=rtc.VideoBufferType.RGB24,
            data=frame_np.tobytes(),
        )

    def _numpy_to_audioframe(self, audio_np: np.ndarray) -> rtc.AudioFrame:
        """
        Convert numpy array to LiveKit AudioFrame.

        Args:
            audio_np: Float32 array in range [-1, 1]

        Returns:
            rtc.AudioFrame (16kHz mono)
        """
        # Convert float32 to int16
        audio_int16 = (audio_np * 32768.0).astype(np.int16)
        audio_int16 = np.clip(audio_int16, -32768, 32767)

        return rtc.AudioFrame(
            data=audio_int16.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(audio_int16),
        )

    async def aclose(self):
        """Cleanup resources gracefully."""
        logger.info("🛑 Closing SyncTalkVideoGenerator...")

        # Signal shutdown
        self._running = False

        # Stop background processor
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            logger.info("Background audio processor stopped")

        # Drain queues
        queues = [
            ("audio_input", self._audio_input_queue),
            ("audio_output", self._audio_output_queue),
            ("video", self._video_queue),
        ]

        for name, q in queues:
            count = 0
            while not q.empty():
                try:
                    q.get_nowait()
                    count += 1
                except:
                    break
            if count > 0:
                logger.info(f"Drained {count} items from {name}_queue")

        # Reset SyncTalk buffers
        self.inference.reset_streaming_buffers()

        logger.info("✅ SyncTalkVideoGenerator closed gracefully")
