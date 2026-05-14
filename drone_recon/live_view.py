#!/usr/bin/env python3
"""
Live View
=========
A standalone OpenCV window that shows what the drone's camera is seeing
in real time, with SAM3's segmentation overlay (mask, bounding box,
centroid, distance label).

Subscribes to BOTH:
  /sam3/debug_image      — annotated frames (SAM3 mask + bbox), ~7 Hz
  /drone/camera/image_raw — raw camera frames, 15 Hz
and renders whichever arrived most recently. That way you get the
overlay when SAM3 just fired and the smooth raw stream in between
inferences — no jittery freeze-frames.

Run in a separate terminal AFTER the simulator is up:

    ros2 run drone_recon live_view

Press 'q' or close the window to exit. Doesn't change anything else
about the running mission.
"""
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class LiveView(Node):
    def __init__(self):
        super().__init__('live_view')

        # Hold the latest frame from each source plus its arrival time so
        # we can render whichever is fresher. The bridge republishes raw
        # camera frames at 15 Hz; SAM3 republishes the same frame with
        # overlay every DETECTION_STRIDE images (~7 Hz) — between
        # inferences the raw stream is more recent.
        self._raw_frame   = None
        self._raw_t       = 0.0
        self._dbg_frame   = None
        self._dbg_t       = 0.0
        self._mission_st  = '—'

        # Track inbound frame rate so we can show it in the title bar
        self._fps_window  = []
        self._fps_max_n   = 30

        self.create_subscription(
            Image, '/drone/camera/image_raw',
            self._cb_raw, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/sam3/debug_image',
            self._cb_debug, qos_profile_sensor_data)
        self.create_subscription(
            String, '/mission/state',
            self._cb_state, 10)

        self._win = 'drone_recon — live view  (press q to close)'
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win, 1280, 720)

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _decode(self, msg: Image):
        if msg.height == 0 or msg.width == 0 or len(msg.data) == 0:
            return None
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        except (ValueError, TypeError):
            return None
        h, w = msg.height, msg.width
        if msg.encoding in ('rgb8',):
            return arr.reshape(h, w, 3)[:, :, ::-1].copy()  # → BGR
        if msg.encoding in ('bgr8',):
            return arr.reshape(h, w, 3).copy()
        return None

    def _cb_raw(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._raw_frame = f
            self._raw_t = time.monotonic()
            self._fps_window.append(self._raw_t)
            if len(self._fps_window) > self._fps_max_n:
                self._fps_window.pop(0)

    def _cb_debug(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._dbg_frame = f
            self._dbg_t = time.monotonic()

    def _cb_state(self, msg):
        self._mission_st = msg.data or '—'

    # ── Render loop ────────────────────────────────────────────────────────

    def _current_fps(self):
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        return (len(self._fps_window) - 1) / span if span > 1e-3 else 0.0

    def _annotate(self, frame, used_dbg: bool):
        """Draw a small overlay strip showing mission state, FPS, and
        whether the frame is raw or comes with the SAM3 overlay."""
        h, _ = frame.shape[:2]
        bar_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (frame.shape[1], h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        text = (f'state: {self._mission_st}  |  '
                f'fps: {self._current_fps():4.1f}  |  '
                f'source: {"sam3 overlay" if used_dbg else "raw camera"}')
        cv2.putText(frame, text, (12, h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                    cv2.LINE_AA)
        return frame

    def render_once(self):
        # Pick the freshest frame. Prefer the SAM3 debug frame within
        # 0.4 s of arrival; older than that, the raw stream is showing
        # newer ground truth and we fall back.
        frame, used_dbg = None, False
        now = time.monotonic()
        if self._dbg_frame is not None and now - self._dbg_t < 0.4:
            frame = self._dbg_frame
            used_dbg = True
        elif self._raw_frame is not None:
            frame = self._raw_frame
        if frame is None:
            return True  # keep waiting

        frame = self._annotate(frame.copy(), used_dbg)
        cv2.imshow(self._win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or Esc
            return False
        # Window-closed-by-X check
        if cv2.getWindowProperty(self._win, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True


def main(args=None):
    rclpy.init(args=args)
    node = LiveView()
    log = node.get_logger()
    log.info('Live view ready — waiting for frames on '
             '/drone/camera/image_raw and /sam3/debug_image')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if not node.render_once():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
