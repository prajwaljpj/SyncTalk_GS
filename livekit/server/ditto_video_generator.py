#!/usr/bin/env python3
"""
Ditto Video Generator - Simplified Implementation
Based on AudioWave pattern from LiveKit examples.

Generates lip-synced video from audio using Ditto TalkingHead SDK.
"""

import os
import sys
import asyncio
import logging
import threading
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Optional

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from stream_pipeline_online import StreamSDK

logger = logging.getLogger(__name__)


class DittoVideoGenerator(VideoGenerator):
    """
    Simplified video generator following AudioWave pattern.

    Key changes from previous implementation:
    - Single input queue (AudioWave pattern)
    - Timeout-based idle generation (no state machine!)
    - Simple video/audio pairing via queues
    - Clean AudioSegmentEnd handling
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        source_path: str,
        *,
        video_width: int = 1280,
        video_height: int = 720,
        video_fps: int = 25,
    ):
        """
        Initialize Ditto video generator.

        Args:
            cfg_pkl: Path to Ditto config pickle
            data_root: Path to Ditto model checkpoints
            source_path: Path to source image
            video_width: Output video width
            video_height: Output video height
            video_fps: Output video FPS
        """
        self._video_width = video_width
        self._video_height = video_height
        self._video_fps = video_fps

        # === AUDIOWAVE PATTERN: Single input queue ===
        self._audio_input_queue: asyncio.Queue[
            rtc.AudioFrame | AudioSegmentEnd
        ] = asyncio.Queue()

        # === DITTO-SPECIFIC: Output queues for pairing ===
        self._video_queue: asyncio.Queue[rtc.VideoFrame] = asyncio.Queue()
        self._audio_output_queue: asyncio.Queue[
            rtc.AudioFrame | AudioSegmentEnd
        ] = asyncio.Queue()

        # === DITTO-SPECIFIC: Audio buffering ===
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = asyncio.Lock()

        # === DITTO-SPECIFIC: Chunk tracking for audio-video pairing ===
        self._chunk_id = 0
        self._pending_audio: dict[int, np.ndarray] = {}  # {chunk_id: audio_samples}
        self._audio_lock = threading.Lock()

        # === Initialize Ditto SDK ===
        logger.info("Initializing Ditto StreamSDK...")
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.sdk.setup(
            source_path,
            output_path="/dev/null",
            frame_callback=self._on_video_frame,  # Ditto's async callback
            online_mode=True,
            fps=25,
        )
        self.sdk.setup_Nd(N_d=1000000)

        # Ditto chunking parameters (from inference.py)
        self.chunksize = (3, 5, 2)  # (past, current, future) frames
        self.split_len = (
            int(sum(self.chunksize) * 0.04 * 16000) + 80
        )  # 6480 samples (matches inference.py)
        self.stride = (
            self.chunksize[1] * 640
        )  # 3200 samples = 5 frames (prevents overlap)

        # Add initial padding - use full chunk size to prevent buffer starvation
        # This ensures buffer always has enough samples for smooth processing
        padding = np.zeros(
            self.split_len, dtype=np.float32
        )  # 6480 samples instead of 1920
        self._audio_buffer = padding.copy()

        # Background task
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"✅ Ditto initialized: {video_width}x{video_height} @ {video_fps}fps"
        )

    # ============================================================================
    # VideoGenerator Interface (Required by LiveKit)
    # ============================================================================

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """
        Push audio from agent to generator.

        This is called by AvatarRunner when agent sends TTS audio.
        AudioWave pattern: Just queue it!
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

        This is called by AvatarRunner when user interrupts.
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

        # Clear pending audio chunks (thread-safe)
        with self._audio_lock:
            self._pending_audio.clear()

        # Reset audio buffer to initial padding
        # Note: Not using async lock since clear_buffer is sync and typically
        # called when processing is paused
        padding = np.zeros(self.split_len, dtype=np.float32)  # Full chunk size
        self._audio_buffer = padding.copy()
        self._chunk_id = 0

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

        # Get the current running event loop
        self._loop = asyncio.get_running_loop()

        self._running = True
        self._processor_task = asyncio.create_task(self._process_audio())
        logger.info("🚀 Background audio processor started")

    async def _process_audio(self):
        """
        Background task: Process incoming audio through Ditto SDK.

        Flow:
        1. Get audio from input queue (with timeout)
        2. Buffer audio samples
        3. When buffer >= chunk size, process through Ditto
        4. Ditto generates video frames via callback
        5. Callback pairs video with corresponding audio

        IMPORTANT: This task ONLY processes TTS audio, NOT idle frames!
        Idle frame generation is handled separately to avoid race conditions.
        """
        while self._running:
            try:
                # Wait for TTS audio from agent
                try:
                    frame = await asyncio.wait_for(
                        self._audio_input_queue.get(), timeout=0.1  # 100ms timeout
                    )
                except asyncio.TimeoutError:
                    # No TTS audio - just wait, don't generate idle frames!
                    # This prevents idle frames from interfering with TTS playback.
                    await asyncio.sleep(0.001)
                    continue

                # Handle AudioSegmentEnd
                if isinstance(frame, AudioSegmentEnd):
                    logger.info("📨 AudioSegmentEnd received - will flush remaining")
                    # Signal end (will be picked up by stream generator)
                    await self._audio_output_queue.put(AudioSegmentEnd())
                    continue

                # Process AudioFrame (TTS only!)
                if isinstance(frame, rtc.AudioFrame):
                    # Resample to 16kHz if needed
                    audio_samples = self._resample_to_16k(frame)
                    logger.info(
                        f"🎵 TTS AUDIO: Received {len(audio_samples)} samples, buffer now {len(self._audio_buffer)} samples"
                    )

                    async with self._buffer_lock:
                        self._audio_buffer = np.concatenate(
                            [self._audio_buffer, audio_samples]
                        )
                        logger.info(
                            f"🎵 Buffer after concat: {len(self._audio_buffer)} samples (need {self.split_len} to process)"
                        )
                        logger.info(
                            f"📊 BUFFERS: audio_buf={len(self._audio_buffer)}, "
                            f"input_q={self._audio_input_queue.qsize()}, "
                            f"output_q={self._audio_output_queue.qsize()}, "
                            f"video_q={self._video_queue.qsize()}, "
                            f"pending_chunks={len(self._pending_audio)}"
                        )

                    # Process TTS chunks through Ditto
                    await self._process_chunks()

            except Exception as e:
                logger.error(f"Audio processor error: {e}", exc_info=True)

    async def _process_chunks(self):
        """Process buffered audio chunks through Ditto SDK."""
        chunks_processed = 0
        async with self._buffer_lock:
            while len(self._audio_buffer) >= self.split_len:
                # Extract chunk with context (past, current, future)
                chunk = self._audio_buffer[: self.split_len].copy()

                # Audio for video (current portion only)
                # Chunk: [past: 1920][current: 3200][future: 1280]
                audio_for_video = chunk[1920:5120]  # 3200 samples = 5 frames worth

                # Store with chunk ID for later pairing
                chunk_id = self._chunk_id
                with self._audio_lock:
                    self._pending_audio[chunk_id] = audio_for_video

                logger.info(
                    f"🎬 CHUNK #{chunk_id}: Sending to Ditto SDK (audio stored: {len(audio_for_video)} samples)"
                )
                logger.info(
                    f"📊 PRE-PROCESS: audio_buf={len(self._audio_buffer)}, pending_chunks={len(self._pending_audio)}"
                )

                self._chunk_id += 1
                chunks_processed += 1

                # Advance buffer by stride (not full chunk!)
                self._audio_buffer = self._audio_buffer[self.stride :]

                # Process through Ditto (blocking call in executor)
                await self._loop.run_in_executor(
                    None, self.sdk.run_chunk, chunk, self.chunksize
                )
                logger.info(
                    f"✅ CHUNK #{chunk_id}: Ditto SDK run_chunk completed (video frames will arrive via callback)"
                )
                logger.info(
                    f"📊 POST-PROCESS: audio_buf={len(self._audio_buffer)}, pending_chunks={len(self._pending_audio)}"
                )
                # Video frames will arrive via _on_video_frame callback

        if chunks_processed > 0:
            logger.info(f"📊 Processed {chunks_processed} chunks in this batch")
            logger.info(
                f"📊 FINAL BUFFERS: audio_buf={len(self._audio_buffer)}, "
                f"output_q={self._audio_output_queue.qsize()}, "
                f"video_q={self._video_queue.qsize()}, "
                f"pending_chunks={len(self._pending_audio)}"
            )

    async def _generate_idle_frame(self) -> tuple[rtc.VideoFrame, rtc.AudioFrame]:
        """
        Generate a single idle frame (neutral expression with silence).
        Called by main loop when no TTS audio is available.

        Optimized to pre-fill buffer efficiently to avoid 1.5s gaps.
        """
        async with self._buffer_lock:
            # Check if buffer needs filling
            if len(self._audio_buffer) < self.split_len:
                # Calculate silence needed to reach chunk threshold
                needed = self.split_len - len(self._audio_buffer)
                # Round up to frame boundary (640 samples)
                frames_needed = (needed + 639) // 640

                # Add ALL needed silence at once (instead of 640 at a time)
                silence = np.zeros(frames_needed * 640, dtype=np.float32)
                self._audio_buffer = np.concatenate([self._audio_buffer, silence])
                logger.debug(
                    f"Pre-filled {frames_needed} frames ({frames_needed * 640} samples) of silence for idle"
                )
                logger.info(
                    f"📊 IDLE PREFILL: audio_buf={len(self._audio_buffer)}, "
                    f"output_q={self._audio_output_queue.qsize()}, "
                    f"video_q={self._video_queue.qsize()}"
                )

            # Process chunk if buffer is ready
            if len(self._audio_buffer) >= self.split_len:
                chunk = self._audio_buffer[: self.split_len].copy()
                audio_for_video = chunk[1920:5120]

                chunk_id = self._chunk_id
                with self._audio_lock:
                    self._pending_audio[chunk_id] = audio_for_video

                self._chunk_id += 1
                self._audio_buffer = self._audio_buffer[self.stride :]

                # Process through Ditto
                await self._loop.run_in_executor(
                    None, self.sdk.run_chunk, chunk, self.chunksize
                )

        # Wait briefly for video frame from callback
        try:
            video = await asyncio.wait_for(self._video_queue.get(), timeout=0.1)
            audio_silence = rtc.AudioFrame(
                data=(np.zeros(640, dtype=np.int16)).tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=640,
            )
            return video, audio_silence
        except asyncio.TimeoutError:
            # If no video available, return None (caller will handle)
            return None, None

    # ============================================================================
    # Ditto SDK Callback (runs in SDK thread)
    # ============================================================================

    def _on_video_frame(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """
        Called by Ditto SDK when video frame is generated.

        Runs in Ditto's worker thread, so must be thread-safe!
        Pairs video frame with corresponding audio and queues both.
        """
        try:
            logger.info(f"🎥 CALLBACK: Received video frame #{frame_idx} from Ditto SDK")

            # Convert BGR to RGB and create VideoFrame
            video_frame = rtc.VideoFrame(
                width=frame_rgb.shape[1],
                height=frame_rgb.shape[0],
                type=rtc.VideoBufferType.RGB24,
                data=frame_rgb.tobytes(),
            )

            # Calculate which chunk this frame belongs to
            chunk_id = frame_idx // 5  # ~5 frames per chunk on average

            # Get corresponding audio
            with self._audio_lock:
                if chunk_id in self._pending_audio:
                    chunk_audio = self._pending_audio[chunk_id]

                    # Extract audio for THIS specific frame
                    # Each frame = 640 samples (40ms @ 16kHz)
                    frame_offset_in_chunk = (frame_idx % 5) * 640

                    if frame_offset_in_chunk + 640 <= len(chunk_audio):
                        audio_samples = chunk_audio[
                            frame_offset_in_chunk : frame_offset_in_chunk + 640
                        ]

                        logger.info(
                            f"🔊 PAIRING: Frame #{frame_idx} (chunk {chunk_id}) paired with {len(audio_samples)} audio samples"
                        )
                        logger.info(
                            f"📊 CALLBACK QUEUES: output_q={self._audio_output_queue.qsize()}, "
                            f"video_q={self._video_queue.qsize()}, "
                            f"pending_chunks={len(self._pending_audio)}"
                        )

                        # Create audio frame
                        audio_int16 = (audio_samples * 32768.0).astype(np.int16)
                        audio_int16 = np.clip(audio_int16, -32768, 32767)
                        audio_frame = rtc.AudioFrame(
                            data=audio_int16.tobytes(),
                            sample_rate=16000,
                            num_channels=1,
                            samples_per_channel=640,
                        )

                        # Queue both (thread-safe)
                        # Queue video FIRST to give it a head start in the scheduling race
                        asyncio.run_coroutine_threadsafe(
                            self._audio_output_queue.put(audio_frame), self._loop
                        )
                        asyncio.run_coroutine_threadsafe(
                            self._video_queue.put(video_frame), self._loop
                        )

                    # Cleanup old chunks to prevent memory leak
                    if chunk_id > 10:
                        self._pending_audio.pop(chunk_id - 10, None)

        except Exception as e:
            logger.error(f"Error in video callback: {e}", exc_info=True)

    # ============================================================================
    # Main Generator (yields frames to AvatarRunner)
    # ============================================================================

    async def _stream_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        """
        Main generator following AudioWave pattern.

        Yields video+audio pairs to AvatarRunner.
        """
        frame_count = 0
        idle_mode = True  # Start in idle mode

        while self._running:  # Check running flag for graceful shutdown
            try:
                # Wait for TTS audio with timeout
                try:
                    audio = await asyncio.wait_for(
                        self._audio_output_queue.get(), timeout=0.05  # 50ms timeout
                    )
                    idle_mode = False  # Got TTS audio
                except asyncio.TimeoutError:
                    # No TTS audio - check if we should generate idle frame
                    if not idle_mode:
                        # First timeout after TTS - switch to idle mode
                        idle_mode = True
                        logger.info("💤 Entering IDLE mode (no TTS audio)")

                    # Generate idle frame
                    video, audio = await self._generate_idle_frame()
                    if video and audio:
                        yield video
                        yield audio
                        frame_count += 1
                    else:
                        # No idle frame ready, just wait
                        await asyncio.sleep(0.001)
                    continue

                # Handle AudioSegmentEnd
                if isinstance(audio, AudioSegmentEnd):
                    logger.info("📤 Yielding AudioSegmentEnd")
                    yield AudioSegmentEnd()
                    idle_mode = True  # Return to idle after segment ends
                    continue

                # Got TTS audio - get corresponding video frame
                if not idle_mode:
                    logger.info("🎤 Entering TTS mode (got audio)")
                    idle_mode = False

                try:
                    video = await asyncio.wait_for(self._video_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Video not ready for audio - skipping")
                    logger.warning(
                        f"📊 TIMEOUT QUEUES: output_q={self._audio_output_queue.qsize()}, "
                        f"video_q={self._video_queue.qsize()}, "
                        f"pending_chunks={len(self._pending_audio)}"
                    )
                    continue

                # Yield synchronized TTS pair
                yield video
                yield audio

                frame_count += 1
                if frame_count % 25 == 0:
                    logger.info(f"📊 Yielded {frame_count} frame pairs")

            except Exception as e:
                if not self._running:
                    # Shutdown in progress, exit gracefully
                    logger.info("Stream generator stopped due to shutdown")
                    break
                logger.error(f"Error in stream generator: {e}", exc_info=True)

        logger.info("Stream generator exited gracefully")

    # ============================================================================
    # Audio Resampling Helper
    # ============================================================================

    def _resample_to_16k(self, frame: rtc.AudioFrame) -> np.ndarray:
        """
        Resample audio frame to 16kHz mono if needed.

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
            import resampy

            audio_data = resampy.resample(
                audio_data, frame.sample_rate, 16000, filter="kaiser_best"
            )

        return audio_data

    async def aclose(self):
        """Cleanup resources gracefully."""
        logger.info("🛑 Closing DittoVideoGenerator...")

        # 1. Signal shutdown to all components
        self._running = False

        # 2. Stop background audio processor
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            logger.info("Background audio processor stopped")

        # 3. Close Ditto SDK (stops all 6 worker threads)
        try:
            logger.info("Closing Ditto SDK worker threads...")
            if hasattr(self, "_loop") and self._loop:
                await self._loop.run_in_executor(None, self.sdk.close)
            else:
                # Fallback if event loop not available
                self.sdk.close()
            logger.info("Ditto SDK closed")
        except Exception as e:
            logger.error(f"Error closing Ditto SDK: {e}", exc_info=True)

        # 4. Drain queues to prevent memory leaks
        queues_to_drain = [
            ("audio_input", self._audio_input_queue),
            ("audio_output", self._audio_output_queue),
            ("video", self._video_queue),
        ]

        total_drained = 0
        for name, q in queues_to_drain:
            count = 0
            while not q.empty():
                try:
                    q.get_nowait()
                    count += 1
                except:
                    break
            if count > 0:
                total_drained += count
                logger.info(f"Drained {count} items from {name}_queue")

        if total_drained > 0:
            logger.info(f"Total items drained: {total_drained}")

        # 5. Clear pending audio chunks (thread-safe)
        with self._audio_lock:
            num_pending = len(self._pending_audio)
            self._pending_audio.clear()
            if num_pending > 0:
                logger.info(f"Cleared {num_pending} pending audio chunks")

        # 6. Clear audio buffer
        async with self._buffer_lock:
            buffer_size = len(self._audio_buffer)
            self._audio_buffer = np.zeros(0, dtype=np.float32)
            if buffer_size > 0:
                logger.info(f"Cleared audio buffer ({buffer_size} samples)")

        # 7. Reset state
        self._chunk_id = 0
        self._frame_idx_counter = 0

        logger.info("✅ DittoVideoGenerator closed gracefully")
