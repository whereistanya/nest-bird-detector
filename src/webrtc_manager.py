"""
WebRTC Stream Manager with Proper Async Threading
Runs asyncio event loop in separate thread to avoid Qt conflicts.

STUN Consent Freshness Limitation:
---------------------------------
The underlying aioice library implements RFC7675 STUN consent freshness checks.
These checks run every 5 seconds at the ICE layer. After 6 consecutive timeouts
(~30 seconds), the connection closes with "Consent to send expired."

This is a known limitation (https://github.com/aiortc/aioice/issues/58) caused by:
1. Python GIL contention delaying STUN response handling in asyncio event loop
2. Zero retransmissions for consent checks - any missed check counts as failure
3. Image processing and Qt GUI can block the GIL, preventing timely responses

Mitigation Strategy:
-------------------
Since we cannot fix this at the library level without forking aioice, we've
implemented seamless automatic reconnection:
- Auto-reconnect when "Consent to send expired" occurs
- Token refresh before each reconnection attempt
- Graceful handling of frame capture during reconnection
- Data channel keepalive (helps but doesn't prevent consent expiration)

Connections typically last 60-90 seconds before requiring reconnection.
Reconnections complete in 3-4 seconds with minimal disruption.
"""
import asyncio
import threading
import queue
from typing import Optional
from queue import Queue
from PIL import Image
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

from sdp_validator import validate_answer_sdp


class WebRTCStreamManager:
    """
    Manages WebRTC connection lifecycle with proper async threading.
    Thread-safe for use in Qt GUI.
    """

    def __init__(self, nest_client):
        self.nest_client = nest_client
        self.pc: Optional[RTCPeerConnection] = None
        self.video_track = None
        self.data_channel = None
        self._connected = False
        self._loop = None
        self._thread = None
        self._latest_frame: Optional[Image.Image] = None
        self._stop_event = threading.Event()
        self._frame_queue = Queue(maxsize=1)
        self._auto_reconnect = False  # Enable automatic reconnection
        self._reconnect_count = 0  # Track reconnection attempts

    async def _start_connection(self) -> bool:
        """Start WebRTC connection asynchronously"""
        try:
            print("🔄 Starting WebRTC stream...")

            # Create peer connection with STUN server
            config = RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
            self.pc = RTCPeerConnection(configuration=config)

            # Track handler
            @self.pc.on("track")
            async def on_track(track):
                print(f"📹 Received {track.kind} track")
                if track.kind == "video":
                    self.video_track = track
                    self._connected = True  # Set connected when video track is ready
                    print("✓ Video track ready")
                    # Start frame capture loop
                    asyncio.create_task(self._frame_capture_loop())

            # Connection state handler
            @self.pc.on("connectionstatechange")
            async def on_state_change():
                state = self.pc.connectionState
                print(f"🔗 Connection state: {state}")
                if state == "connected":
                    self._connected = True
                    self._reconnect_count = 0  # Reset on successful connection
                elif state in ["failed", "closed"]:
                    self._connected = False
                    # Trigger automatic reconnection if enabled and not manually stopped
                    if self._auto_reconnect and not self._stop_event.is_set():
                        self._reconnect_count += 1
                        print(f"🔄 Stream closed. Auto-reconnecting... (attempt {self._reconnect_count})")
                        # Wait a bit before reconnecting
                        await asyncio.sleep(2.0)
                        if not self._stop_event.is_set():
                            # Trigger reconnection by scheduling it in the event loop
                            asyncio.create_task(self._reconnect())

            # Add transceivers for audio and video (receive-only)
            self.pc.addTransceiver("audio", direction="recvonly")
            self.pc.addTransceiver("video", direction="recvonly")

            # Add data channel (required for application m-line and keepalive)
            self.data_channel = self.pc.createDataChannel("keepalive")

            # Data channel handlers
            @self.data_channel.on("open")
            def on_datachannel_open():
                print("📡 Data channel open - starting keepalive")
                # Start keepalive task
                asyncio.create_task(self._keepalive_loop())

            @self.data_channel.on("close")
            def on_datachannel_close():
                print("📡 Data channel closed")

            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            print("✓ Created local offer")

            # Get SDP offer to send to Nest
            offer_sdp = self.pc.localDescription.sdp

            # Request stream from Nest with our offer
            answer_sdp, session_id = self.nest_client.generate_webrtc_stream(offer_sdp=offer_sdp)
            print(f"✓ Got WebRTC answer (session: {session_id[:20]}...)")

            # VALIDATE SDP BEFORE USING IT
            is_valid, error = validate_answer_sdp(answer_sdp)
            if not is_valid:
                print(f"❌ SDP validation failed: {error}")
                return False

            print("✓ SDP answer validated")

            # Fix Nest's SDP format for aiortc compatibility
            fixed_lines = []
            for line in answer_sdp.split('\n'):
                if line.startswith('a=candidate:'):
                    # Remove extra space
                    line = line.replace('a=candidate: ', 'a=candidate:')
                    # Insert component ID (always 1 for RTP)
                    parts = line.split(' ')
                    if len(parts) >= 2 and parts[1] in ['udp', 'tcp', 'ssltcp']:
                        # Missing component ID, insert it
                        parts.insert(1, '1')
                        line = ' '.join(parts)
                fixed_lines.append(line)
            answer_sdp = '\n'.join(fixed_lines)
            print("✓ Fixed SDP format")

            # Set remote description (Nest's answer)
            await self.pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer_sdp, type="answer")
            )

            print("✓ Set remote description")

            # Wait for connection (max 15 seconds)
            print("⏳ Waiting for connection...")
            for i in range(150):
                if self._connected and self.video_track:
                    print("✅ WebRTC stream connected and ready!")
                    return True
                await asyncio.sleep(0.1)
                if i % 10 == 0 and i > 0:
                    print(f"   Still waiting... ({i/10:.0f}s)")

            print("❌ Connection timeout")
            return False

        except Exception as e:
            print(f"❌ WebRTC connection failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _frame_capture_loop(self):
        """Continuously capture frames from video track"""
        print("▶️  Frame capture loop started")
        error_count = 0
        max_errors = 5
        frame_count = 0
        corrupted_frame_count = 0

        while not self._stop_event.is_set() and self._connected and self.video_track:
            try:
                # Receive frame
                frame = await self.video_track.recv()

                # Convert to RGB numpy array
                img = frame.to_ndarray(format="rgb24")

                # Basic frame validation to detect severely corrupted frames
                import numpy as np
                if img.size > 0:
                    # Sample-based corruption check: pick a sparse grid of pixels
                    # and check if they're all the same color (much cheaper than np.unique on full frame)
                    h, w = img.shape[:2]
                    sample_rows = np.linspace(0, h - 1, min(20, h), dtype=int)
                    sample_cols = np.linspace(0, w - 1, min(20, w), dtype=int)
                    sample = img[np.ix_(sample_rows, sample_cols)]
                    unique_samples = len(np.unique(sample.reshape(-1, sample.shape[2]), axis=0))

                    if unique_samples < 3:
                        corrupted_frame_count += 1
                        # Skip this frame, wait for next one
                        if corrupted_frame_count % 50 == 0:  # Log every 50th (less noisy)
                            print(f"⚠️  Skipped {corrupted_frame_count} corrupted frames")
                        await asyncio.sleep(0.01)
                        continue

                # Convert to PIL Image
                pil_image = Image.fromarray(img)

                # Update queue (drop old frame if full)
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass  # Queue was already empty
                self._frame_queue.put(pil_image)

                frame_count += 1

                # Reset error count on success
                error_count = 0

            except Exception as e:
                error_count += 1
                if not self._stop_event.is_set():
                    # Only print error for non-empty errors
                    error_msg = str(e).strip()
                    if error_msg:
                        print(f"⚠️  Frame capture error ({error_count}/{max_errors}): {type(e).__name__}: {e}")
                    else:
                        print(f"⚠️  Frame capture error ({error_count}/{max_errors}): {type(e).__name__}")

                # Only break after multiple consecutive errors
                if error_count >= max_errors:
                    print("❌ Too many frame capture errors, stopping loop")
                    break

                # Wait a bit before retrying
                await asyncio.sleep(0.1)

        if corrupted_frame_count > 0:
            print(f"⏹️  Frame capture loop stopped ({frame_count} frames captured, {corrupted_frame_count} corrupted frames skipped)")
        else:
            print(f"⏹️  Frame capture loop stopped ({frame_count} frames captured)")

    async def _keepalive_loop(self):
        """Send periodic keepalive messages to prevent DTLS consent expiration"""
        print("💓 Keepalive loop started")
        keepalive_count = 0

        while not self._stop_event.is_set() and self._connected and self.data_channel:
            try:
                # Check if data channel is still open
                if self.data_channel.readyState == "open":
                    # Send keepalive message (every 15 seconds to stay well below 30s timeout)
                    self.data_channel.send("keepalive")
                    keepalive_count += 1
                    if keepalive_count % 4 == 0:  # Log every minute (4 * 15s)
                        print(f"💓 Keepalive sent ({keepalive_count} total)")
                else:
                    # Data channel closed, stop keepalive
                    print("💓 Data channel not open, stopping keepalive")
                    break

                # Wait 15 seconds before next keepalive
                await asyncio.sleep(15.0)

            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"⚠️  Keepalive error: {type(e).__name__}: {e}")
                break

        print(f"💓 Keepalive loop stopped ({keepalive_count} keepalives sent)")

    async def _stop_connection(self):
        """Stop WebRTC connection"""
        self._connected = False

        # Close data channel
        if self.data_channel:
            try:
                self.data_channel.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self.data_channel = None

        if self.pc:
            await self.pc.close()
            self.pc = None

        self.video_track = None

    async def _reconnect(self):
        """Reconnect WebRTC stream after disconnection"""
        try:
            # Close old connection
            if self.pc:
                await self._stop_connection()

            # Ensure token is refreshed before reconnecting
            # This prevents token expiration during auto-reconnect
            try:
                self.nest_client._ensure_valid_token()
                print("✓ Token validated/refreshed for reconnection")
            except Exception as token_error:
                print(f"❌ Token refresh failed during reconnect: {token_error}")
                print("   Cannot reconnect with expired token. Auto-reconnect will retry later.")
                # Don't attempt connection with expired token
                return

            # Start new connection
            success = await self._start_connection()
            if not success:
                print("❌ Reconnection failed")
        except Exception as e:
            print(f"❌ Reconnection error: {e}")

    def _run_event_loop(self, connection_result_queue):
        """Run async event loop in separate thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            # Start connection
            success = self._loop.run_until_complete(self._start_connection())
            connection_result_queue.put(success)

            # Keep loop running for frame capture
            if success:
                self._loop.run_until_complete(self._keep_alive())
        except Exception as e:
            print(f"Event loop error: {e}")
            connection_result_queue.put(False)
        finally:
            # Cleanup
            if self.pc:
                self._loop.run_until_complete(self._stop_connection())
            self._loop.close()

    async def _keep_alive(self):
        """Keep event loop alive"""
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)

    # Public methods for GUI

    def start_stream(self) -> bool:
        """
        Start WebRTC stream (synchronous, GUI-safe).
        Returns True if successful.
        """
        self._stop_event.clear()
        self._auto_reconnect = True  # Enable automatic reconnection
        self._reconnect_count = 0
        connection_result_queue = Queue()

        # Start async event loop in separate thread
        self._thread = threading.Thread(
            target=self._run_event_loop,
            args=(connection_result_queue,),
            daemon=True
        )
        self._thread.start()

        # Wait for connection result (max 20 seconds)
        try:
            success = connection_result_queue.get(timeout=20)
            return success
        except Exception:
            print("❌ Connection timeout")
            return False

    def capture_frame(self) -> Optional[Image.Image]:
        """
        Get latest captured frame (synchronous, GUI-safe).
        Returns PIL Image or None.
        """
        if not self._connected:
            return None

        try:
            # Get frame from queue with short timeout for responsiveness
            return self._frame_queue.get(timeout=0.5)
        except queue.Empty:
            return None  # No frame available within timeout

    def stop_stream(self):
        """Stop WebRTC stream (synchronous, GUI-safe)"""
        self._stop_event.set()
        self._auto_reconnect = False  # Disable automatic reconnection
        self._connected = False

        if self._thread and self._thread.is_alive():
            # Give thread up to 5 seconds to cleanly shut down
            self._thread.join(timeout=5.0)

            # Warn if thread didn't terminate properly
            if self._thread.is_alive():
                print("⚠️  Warning: WebRTC thread did not terminate within 5s timeout")
                print("   Stream may not have closed cleanly")

        self._thread = None
        self._loop = None

    def is_connected(self) -> bool:
        """Check if stream is connected"""
        return self._connected

