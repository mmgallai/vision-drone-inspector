#!/usr/bin/env python3
"""
SAM3 Detector Node
==================
Uses Meta's Segment Anything Model 3 (SAM3) via Ultralytics to detect
and segment the target object in the drone's camera feed using a text prompt.
No training or fiducial markers required — zero-shot detection.

Topics published:
  /sam3/detected     (std_msgs/Bool)     - True when target is visible
  /sam3/debug_image  (sensor_msgs/Image) - annotated frame with mask overlay
  /sam3/centroid_x   (std_msgs/Float32)  - target centroid X, normalized [0,1]
  /sam3/centroid_y   (std_msgs/Float32)  - target centroid Y, normalized [0,1]
  /sam3/distance     (std_msgs/Float32)  - estimated distance to target (meters)

Topics subscribed:
  /drone/camera/image_raw  (sensor_msgs/Image)        - drone camera feed
  /drone/pose              (geometry_msgs/PoseStamped) - drone world pose
  /mission/state           (std_msgs/String)           - current mission state

Detection strategy:
  SEARCH / APPROACH  → SAM3 runs every DETECTION_STRIDE frames (~800ms inference)
  ORBIT_LOW / HIGH   → OpenCV CSRT tracker (~1ms), SAM3 only on tracker loss
"""

import os
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import PoseStamped
# cv_bridge is NOT used here — its C extension is compiled against numpy 1.x
# and fails with our numpy 2.x PYTHONPATH injection. Images are decoded manually.

from ultralytics.models.sam import SAM, SAM3SemanticPredictor
from ultralytics.engine.model import Model as _UltralyticsModel

from drone_recon import targets as _targets


class SAM3Semantic(SAM):
    """
    SAM subclass wired to SAM3SemanticPredictor so that text-prompted
    (zero-shot) segmentation works with the facebook/sam3 weights.

    The stock SAM class selects SAM3Predictor (interactive / geometric) for
    any file whose stem contains 'sam3', but the facebook/sam3 checkpoint is
    for the semantic (vision-language) architecture.  The two differences:
      1. _load  → calls build_sam3_image_model instead of build_interactive_sam3
      2. task_map → returns SAM3SemanticPredictor
      3. predict → passes 'text' via the prompts dict (the only path that
                   reaches SAM3SemanticPredictor.inference's text= parameter)
    """

    def _load(self, weights: str, task=None):
        from ultralytics.models.sam.build_sam3 import build_sam3_image_model
        self.model = build_sam3_image_model(weights)

    @property
    def task_map(self):
        return {'segment': {'predictor': SAM3SemanticPredictor}}

    def predict(self, source, stream=False, text=None, **kwargs):
        prompts = {'text': text} if text else {}
        return _UltralyticsModel.predict(self, source, stream,
                                         prompts=prompts, **kwargs)


# ── Parameters ────────────────────────────────────────────────────────────────

# SAM3 stride: run every Nth frame during search/approach (~800ms inference)
DETECTION_STRIDE = 2

# Keep detected=True for this many seconds after the last positive hit.
STICKY_SECS = 2.0

# States in which we use the tracker instead of SAM3
TRACKER_STATES = {'ORBIT_LOW', 'ORBIT_HIGH', 'CLIMB'}

# How many consecutive tracker failures before we fall back to SAM3
TRACKER_FAIL_LIMIT = 5

# message_filters sync window. RGB at 15Hz, depth at 10Hz, pose at 20Hz
# means the worst-case gap between an RGB frame and the nearest depth
# frame is ~50ms. 100ms slop is comfortable headroom.
SYNC_QUEUE_SIZE = 10
SYNC_SLOP_SECS  = 0.10

# RGB camera: must match SDF (1280×720, hfov=90°)
IMG_W, IMG_H = 1280, 720
FX = IMG_W / (2.0 * np.tan(np.radians(45.0)))  # = 640.0 px

# Depth camera: must match SDF (640×480, hfov=90°, near 0.05, far 20)
DEPTH_W, DEPTH_H = 640, 480
DEPTH_FX = DEPTH_W / 2.0      # = 320 px (hfov=90 → fx = w/2)
DEPTH_FAR = 20.0
# Practical near clip: anything closer than this is implausible for an
# in-flight drone at search/orbit altitude (1.5+ m). Raising the floor
# from the SDF's 0.05 m to 0.3 m filters out the noise spikes that gave
# every SAM3 first-detection a bogus "dist=0.11 m" — those values
# back-projected to wild XY estimates that the mission-side hit-validator
# had to throw away. With this filter the FIRST hit is already clean.
DEPTH_NEAR = 0.3
DEPTH_PATCH_HALF = 5          # 11×11 sample patch around the centroid


# ── Node ──────────────────────────────────────────────────────────────────────

class SAM3DetectorNode(Node):
    """
    Runs SAM3 on the drone camera stream.
    Detects the target via text prompt and publishes segmentation results.

    Text-prompted segmentation means:
      - We describe the target in plain English (e.g. "fire hydrant")
      - SAM3 segments ALL matching instances in the frame
      - We pick the largest mask as the primary target
    """

    def __init__(self):
        super().__init__('sam3_detector')

        # ── ROS Parameters ──────────────────────────────────────────────
        self.declare_parameter('target_prompt', 'fire hydrant')
        self.declare_parameter('model_size',    'b')   # 'b' (base) or 'l' (large)
        self.declare_parameter('confidence',    0.35)
        # Default lives in the user's home (~/sam3/) so it survives a
        # colcon `rm -rf build/`. Override via the launch arg or
        # SAM3_WEIGHTS_PATH env var. See docs/sam3_install.md.
        self.declare_parameter('model_path',
            os.path.expanduser('~/sam3/sam3.pt'))

        prompt     = self.get_parameter('target_prompt').value
        model_size = self.get_parameter('model_size').value
        self.conf  = self.get_parameter('confidence').value

        # Normalize prompt and pull target config (height, prune box, etc.)
        # from the shared targets table.
        self.target_prompt = prompt.strip()
        self.target_cfg    = _targets.get(self.target_prompt)
        self.target_height = self.target_cfg['height_m']

        # ── Load SAM3 ───────────────────────────────────────────────────
        # Path is configurable via the 'model_path' ROS parameter so it works
        # both on the host (~/sam3/...) and inside Docker (/root/sam3/...).
        sam3_path = self.get_parameter('model_path').value
        # Pre-flight: weights file must exist. The original failure mode
        # was torch.load() crashing 3 s into startup with a generic
        # FileNotFoundError that was easy to miss in the launch noise.
        # Now we log a clear, actionable message and raise SystemExit so
        # the launch supervisor reports the node as died-cleanly with a
        # useful explanation right above it.
        if not os.path.isfile(sam3_path):
            self.get_logger().error(
                f'SAM3 weights not found at: {sam3_path}\n'
                f'Re-download instructions: see docs/sam3_install.md\n'
                f'(or pass `model_path:=...` ROS param / SAM3_WEIGHTS_PATH '
                f'env var to use a different location).')
            raise SystemExit(2)
        self.get_logger().info(f'Loading SAM3 from {sam3_path} ...')
        self.model = SAM3Semantic(sam3_path)
        self.model.to('cuda')
        self.get_logger().info(
            f'SAM3 ready  |  target: "{self.target_prompt}"  '
            f'|  target height: {self.target_height}m')

        # ── Publishers ──────────────────────────────────────────────────
        self.pub_detected   = self.create_publisher(Bool,   '/sam3/detected',    10)
        self.pub_debug      = self.create_publisher(Image,  '/sam3/debug_image', 10)
        self.pub_centroid_x = self.create_publisher(Float32,'/sam3/centroid_x',  10)
        self.pub_centroid_y = self.create_publisher(Float32,'/sam3/centroid_y',  10)
        self.pub_distance   = self.create_publisher(Float32,'/sam3/distance',    10)
        # SAM3's per-mask confidence for the prompted class. Used by
        # mission_node to disambiguate between competing clusters of
        # detections — when SAM3 is more confident the object IS the
        # prompted thing, the cluster centroid is more trustworthy.
        self.pub_score      = self.create_publisher(Float32,'/sam3/score',       10)

        # ── Subscribers ─────────────────────────────────────────────────
        # Plain "latest-of" subscribers. We previously used a
        # message_filters ApproximateTimeSynchronizer here for tighter
        # image↔depth↔pose alignment, but it silently failed to assemble
        # triplets when sim-time gz stamps had to be matched against
        # wall-time stamps from mission_node — SAM3 then never received
        # any frames at all and the drone fell back to its known-position
        # path on every mission. The old "snapshot depth at SAM3 input
        # time" trick is good enough.
        self.sub_image = self.create_subscription(
            Image, '/drone/camera/image_raw', self._cb_image, 10)
        self.sub_depth = self.create_subscription(
            Image, '/drone/depth_camera/image_raw', self._cb_depth, 10)
        self.sub_pose = self.create_subscription(
            PoseStamped, '/drone/pose', self._cb_pose, 10)
        self.sub_state = self.create_subscription(
            String, '/mission/state', self._cb_mission_state, 10)

        # ── Internal state ──────────────────────────────────────────────
        self.frame_count       = 0
        self.drone_z           = 0.0
        self._detected         = False
        self._last_detect_time = 0.0
        self._total_detections = 0
        # Latest depth image as float32 (H, W) array, plus the depth frame
        # captured at SAM3 input time (so we sample the depth that matches
        # the RGB SAM3 actually saw — the drone moves during inference).
        self._latest_depth     = None
        self._depth_for_sam3   = None
        # Counters for distance-source telemetry
        self._n_dist_depth     = 0
        self._n_dist_bbox      = 0

        # ── Tracker state ────────────────────────────────────────────────
        self._mission_state    = 'SEARCH'
        self._tracker          = None   # OpenCV CSRT tracker instance
        self._tracker_failures = 0      # consecutive frames tracker lost target

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _cb_pose(self, msg: PoseStamped):
        """Latest-of pose. Used for the drone_z field which feeds into
        the bbox-height fallback distance estimate."""
        self.drone_z = msg.pose.position.z

    def _cb_depth(self, msg: Image):
        """Latest-of depth image. Stashed here so _depth_distance() can
        sample it when SAM3 returns a centroid. We snapshot this into
        _depth_for_sam3 just before SAM3 inference so the depth used for
        distance estimation matches the RGB SAM3 actually saw."""
        decoded = self._decode_depth(msg)
        if decoded is not None:
            self._latest_depth = decoded

    @staticmethod
    def _decode_depth(msg: Image):
        """Decode a 32FC1 depth image as an (H,W) float32 numpy array.
        Returns None on shape/encoding error."""
        if msg.height == 0 or msg.width == 0 or len(msg.data) == 0:
            return None
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
            return arr.reshape(msg.height, msg.width)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _rgb_norm_to_depth_pixel(cx_norm: float, cy_norm: float):
        """Map a SAM3 mask centroid in normalized RGB coords [0,1]² to a
        pixel in the depth image. Both cameras share an optical center and
        hfov=90°, so horizontal coords scale linearly. Vertically the depth
        camera sees a wider FOV (480/320 vs 720/640), so we recover the
        common ray angle and reproject it through the depth camera's
        focal length.
            angle_y = (ry_rgb - cy_rgb) / fy_rgb
            ry_depth = cy_depth + angle_y * fy_depth
        """
        # Horizontal: same hfov → preserve normalized X
        dx = int(round(cx_norm * DEPTH_W))
        # Vertical: ry_rgb in pixels → angle → depth pixel
        ry_rgb_px = cy_norm * IMG_H
        angle_y = (ry_rgb_px - IMG_H / 2.0) / FX        # fx == fy for square pixels
        dy = int(round(DEPTH_H / 2.0 + angle_y * DEPTH_FX))
        # Clamp inside the depth image
        dx = max(0, min(DEPTH_W - 1, dx))
        dy = max(0, min(DEPTH_H - 1, dy))
        return dx, dy

    @staticmethod
    def _rgb_mask_to_depth_mask(mask_rgb):
        """Warp an (IMG_H, IMG_W) uint8 binary mask into the depth image's
        coordinate frame using the same RGB↔depth pixel relation as
        `_rgb_norm_to_depth_pixel` — but applied to every mask pixel at
        once via a 2x3 affine transform.

            x_d = sx * x_rgb
            y_d = ty + sy * y_rgb
        where sx = DEPTH_W / IMG_W, sy = DEPTH_FX / FX,
        ty = DEPTH_H/2 - sy * IMG_H/2.

        Nearest-neighbor preserves the binary mask. Pixels that fall
        outside the depth image (top/bottom strips that depth sees but
        RGB doesn't) just become 0 — they were never part of the mask
        anyway."""
        sx = DEPTH_W / IMG_W
        sy = DEPTH_FX / FX
        ty = DEPTH_H / 2.0 - sy * IMG_H / 2.0
        M = np.array([[sx, 0.0, 0.0],
                      [0.0, sy, ty]], dtype=np.float32)
        return cv2.warpAffine(mask_rgb, M, (DEPTH_W, DEPTH_H),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _depth_distance_under_mask(self, mask_rgb,
                                   cx_norm: float, cy_norm: float) -> float:
        """
        True metric distance to the segmented object, computed by sampling
        the depth image at EVERY pixel under the SAM3 mask (after erosion
        to drop edge bleed-through to background).

        This is the difference between visual identification and actual
        localization: SAM3 says "the plant is these pixels"; the depth
        camera tells us how far each of those pixels is. Erosion +
        low-percentile aggregation is robust to a few stray near/far
        samples (mask edges spilling onto a wall behind the object,
        foliage gaps, sensor noise on transparent leaves).

        Returns slant range along the centroid ray, or NaN if too few
        samples landed inside the depth image's valid range.
        """
        depth = self._depth_for_sam3
        if depth is None or mask_rgb is None:
            return float('nan')

        mask_d = self._rgb_mask_to_depth_mask(mask_rgb)
        # Erode the mask by a few pixels to drop the silhouette boundary
        # where rasterized edges include both foreground and background.
        # SAM3 masks are typically a few pixels too generous around the
        # contour — those edge pixels read background-distance and pull
        # the percentile away from the actual object surface.
        erode_k = np.ones((5, 5), np.uint8)
        mask_d = cv2.erode(mask_d, erode_k, iterations=2).astype(bool)
        if not mask_d.any():
            # Erosion ate the whole mask (small/thin object). Fall back
            # to the un-eroded version so we don't lose the detection.
            mask_d = self._rgb_mask_to_depth_mask(mask_rgb).astype(bool)
        if not mask_d.any():
            return float('nan')

        samples = depth[mask_d]
        valid = samples[(samples > DEPTH_NEAR) & (samples < DEPTH_FAR)
                        & np.isfinite(samples)]
        # Need a meaningful number of samples — small or fragmented masks
        # under heavy occlusion can still leave us with too little signal.
        if valid.size < 50:
            return float('nan')

        # Diagnostic — log the depth distribution under the mask for the
        # first few hits so we can sanity-check the percentile choice.
        if self._total_detections < 5:
            try:
                self.get_logger().info(
                    f' [SAM3:depth-diag] valid={valid.size} '
                    f'min={float(valid.min()):.2f} '
                    f'p10={float(np.percentile(valid,10)):.2f} '
                    f'p30={float(np.percentile(valid,30)):.2f} '
                    f'p50={float(np.median(valid)):.2f} '
                    f'p70={float(np.percentile(valid,70)):.2f} '
                    f'max={float(valid.max()):.2f}')
            except Exception:
                pass
        # 10th percentile, biased hard toward the foreground surface.
        # A SAM3 mask drawn around a leafy plant (or any object with
        # silhouette gaps) covers (a) the actual object surface and
        # (b) background bleed-through wherever the mask is loose or
        # the foliage has holes. Background pixels report depth to the
        # wall behind the object — many meters too far. Even the median
        # or 30th-percentile can be polluted when gaps dominate.
        # 10th percentile picks the closest 10% of mask pixels, which
        # is virtually always actual object surface (the plant pot,
        # canopy edge, etc.) even when half the mask is bleed-through.
        # We still drop the absolute minimum to avoid single-pixel
        # depth-camera noise spikes.
        z = float(np.percentile(valid, 10))
        # Convert Z (optical-axis distance) to slant range along the
        # centroid ray, since mission_node consumes `dist` as the
        # along-ray range used by ray-cast localization.
        dx, dy = self._rgb_norm_to_depth_pixel(cx_norm, cy_norm)
        rx = (dx - DEPTH_W / 2.0) / DEPTH_FX
        ry = (dy - DEPTH_H / 2.0) / DEPTH_FX  # square pixels: fy == fx
        return z * float(np.sqrt(rx * rx + ry * ry + 1.0))

    def _depth_distance_centroid_patch(self, cx_norm: float,
                                       cy_norm: float) -> float:
        """Centroid-patch fallback used when no mask is available (the
        OpenCV tracker only provides a bbox). Same math as
        `_depth_distance_under_mask` but samples a small patch around the
        centroid rather than every mask pixel — coarser, but workable."""
        depth = self._depth_for_sam3
        if depth is None:
            return float('nan')

        dx, dy = self._rgb_norm_to_depth_pixel(cx_norm, cy_norm)
        H, W = depth.shape
        x0, x1 = max(0, dx - DEPTH_PATCH_HALF), min(W, dx + DEPTH_PATCH_HALF + 1)
        y0, y1 = max(0, dy - DEPTH_PATCH_HALF), min(H, dy + DEPTH_PATCH_HALF + 1)
        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > DEPTH_NEAR) & (patch < DEPTH_FAR) & np.isfinite(patch)]
        if valid.size < max(3, patch.size // 4):
            return float('nan')

        z = float(np.median(valid))
        rx = (dx - DEPTH_W / 2.0) / DEPTH_FX
        ry = (dy - DEPTH_H / 2.0) / DEPTH_FX
        return z * float(np.sqrt(rx * rx + ry * ry + 1.0))

    def _cb_mission_state(self, msg: String):
        prev = self._mission_state
        self._mission_state = msg.data
        # Reset tracker when leaving orbit (e.g. re-running the mission)
        if prev in TRACKER_STATES and msg.data not in TRACKER_STATES:
            self._tracker = None
            self._tracker_failures = 0

    def _cb_image(self, img_msg: Image):
        """
        RGB callback. Drives SAM3 (or the lightweight tracker, in orbit
        states). Pose and depth come in via separate subscribers and we
        snapshot them at SAM3-input time.
        """
        self.frame_count += 1

        # Guard: skip empty/zero-dimension frames
        if len(img_msg.data) == 0 or img_msg.width == 0 or img_msg.height == 0:
            return

        # Decode image manually (avoids cv_bridge numpy 1.x/2.x ABI conflict).
        # ros_gz_bridge publishes rgb8; we need BGR for OpenCV/SAM3.
        raw = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)
        frame = raw.reshape(img_msg.height, img_msg.width, 3)[:, :, ::-1].copy()

        # Snapshot the depth that was current as we received this image,
        # so _depth_distance() samples something close to what SAM3 sees.
        self._depth_for_sam3 = self._latest_depth

        use_tracker = (
            self._mission_state in TRACKER_STATES
            and self._tracker is not None
            and self._tracker_failures < TRACKER_FAIL_LIMIT
        )

        if use_tracker:
            detected, cx, cy, dist, debug = self._track(frame)
            elapsed_ms = 0.0
            mode = 'TRK'
            score = 0.0
        else:
            # SAM3 stride only applies outside orbit (during orbit we only
            # reach here when the tracker is absent or has failed)
            if (self._mission_state not in TRACKER_STATES
                    and self.frame_count % DETECTION_STRIDE != 0):
                return

            t0 = time.monotonic()
            results = self.model.predict(
                source=frame,
                text=[self.target_prompt],
                conf=self.conf,
                imgsz=644,
                verbose=False,
                device='cuda',
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            detected, cx, cy, dist, debug, bbox, score = self._parse(results, frame)

            # Initialize / re-initialize tracker on every successful SAM3 hit
            if detected and bbox is not None:
                self._tracker = cv2.TrackerMIL_create()
                self._tracker.init(frame, bbox)
                self._tracker_failures = 0
            mode = 'SAM3'

            # Dump the first N annotated detections to disk so we can
            # visually verify SAM3 is segmenting the prompted object and
            # not a wall/bench/etc that looks superficially similar.
            # Limited to the first 10 hits to avoid filling the disk.
            if detected and self._total_detections < 10:
                try:
                    os.makedirs('/tmp/sam3_dbg', exist_ok=True)
                    out = (f'/tmp/sam3_dbg/'
                           f'hit_{self._total_detections + 1:02d}.png')
                    cv2.imwrite(out, debug)
                except OSError:
                    pass

        # Sticky flag logic
        now = time.monotonic()
        if detected:
            self._last_detect_time = now
            if not self._detected:
                self._total_detections += 1
                total = self._n_dist_depth + self._n_dist_bbox
                src_mix = (f'  depth/bbox={self._n_dist_depth}/{self._n_dist_bbox}'
                           if total > 0 else '')
                self.get_logger().info(
                    f' [{mode}] #{self._total_detections} detected '
                    f'"{self.target_prompt}"  '
                    f'dist={dist:.2f}m  cx={cx:.2f} cy={cy:.2f}  '
                    f'score={score:.2f}'
                    + (f'  inference={elapsed_ms:.0f}ms' if elapsed_ms else '')
                    + src_mix)
            self._detected = True
        else:
            if now - self._last_detect_time > STICKY_SECS:
                self._detected = False

        self._publish(img_msg.header, detected, debug, cx, cy, dist, score)

    # ──────────────────────────────────────────────────────────────────────
    # Result Parsing
    # ──────────────────────────────────────────────────────────────────────

    def _parse(self, results, frame):
        """
        Extract the best mask from SAM3 results.
        Returns: (detected, cx_norm, cy_norm, distance_m, debug_frame,
                  bbox_xywh, score)
        bbox_xywh is the bounding rect (x,y,w,h) for tracker init, or None.
        score is SAM3's per-mask confidence for the prompted class
        (best mask only), or 0.0 if unavailable.
        """
        h, w = frame.shape[:2]
        debug = frame.copy()
        cx_norm = cy_norm = 0.5
        dist = 0.0
        bbox = None
        score = 0.0

        # Diagnostic — periodically log what SAM3 returned (or didn't)
        # so we can tell whether SAM3 is failing because (a) it returned
        # zero masks, or (b) the masks were too small/empty after our
        # filtering. Log throttled to ~1 Hz so it doesn't flood at 7-8 Hz.
        n_masks = (results[0].masks.data.shape[0]
                   if results and results[0].masks is not None else 0)
        if n_masks == 0:
            self.get_logger().info(
                f' [SAM3:diag] no masks for "{self.target_prompt}" '
                f'(conf={self.conf})',
                throttle_duration_sec=1.0)
        else:
            areas = [int(m.sum().item()) for m in results[0].masks.data]
            self.get_logger().info(
                f' [SAM3:diag] {n_masks} masks for "{self.target_prompt}" '
                f'areas={areas[:5]}',
                throttle_duration_sec=1.0)

        if not results or results[0].masks is None:
            return False, cx_norm, cy_norm, dist, debug, bbox, score

        masks_tensor = results[0].masks.data
        if masks_tensor.shape[0] == 0:
            return False, cx_norm, cy_norm, dist, debug, bbox, score

        # Pick the mask SAM3 itself thinks is most likely the prompted
        # object (highest score), NOT just the largest one. With multiple
        # plausible matches in frame (e.g. SAM3 occasionally segmenting
        # both a real "potted plant" and a similarly-shaped park bench),
        # picking by area would lock onto whichever happens to fill more
        # pixels — which depends on the drone's vantage point, not the
        # object's identity. Confidence is what actually says "this is
        # the prompted thing".
        try:
            scores = results[0].boxes.conf.cpu().numpy()
        except Exception:
            scores = None
        if scores is not None and len(scores) == masks_tensor.shape[0]:
            best_i = int(np.argmax(scores))
            score = float(scores[best_i])
        else:
            # Fallback to area if scores aren't available for any reason.
            areas = [m.sum().item() for m in masks_tensor]
            best_i = int(np.argmax(areas))
        mask_np = masks_tensor[best_i].cpu().numpy().astype(np.uint8)

        if mask_np.shape != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        M = cv2.moments(mask_np)
        if M['m00'] < 1:
            return False, cx_norm, cy_norm, dist, debug, bbox, score

        cx_px = M['m10'] / M['m00']
        cy_px = M['m01'] / M['m00']
        cx_norm = cx_px / w
        cy_norm = cy_px / h

        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            bbox = (x, y, bw, bh)

            # Primary: median depth-camera reading across every pixel
            # under the SAM3 mask. This gives true metric distance to
            # the segmented object (no centroid-on-background bug).
            # Fallback: bbox height + assumed real-world target height,
            # used only when the mask warps outside the depth FOV or the
            # depth values under it are all invalid.
            dist_src = 'bbox'
            d_depth = self._depth_distance_under_mask(mask_np, cx_norm, cy_norm)
            if np.isfinite(d_depth) and d_depth > DEPTH_NEAR:
                dist = d_depth
                dist_src = 'depth'
                self._n_dist_depth += 1
            elif bh > 5:
                dist = (self.target_height * FX) / bh
                self._n_dist_bbox += 1

            overlay = debug.copy()
            overlay[mask_np == 1] = (0, 200, 80)
            cv2.addWeighted(overlay, 0.4, debug, 0.6, 0, debug)
            cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(debug, (int(cx_px), int(cy_px)), 7, (0, 255, 255), -1)
            label = f'[SAM3:{dist_src}] {self.target_prompt}  {dist:.1f}m'
            cv2.putText(debug, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        return True, cx_norm, cy_norm, dist, debug, bbox, score

    def _track(self, frame):
        """
        Run one step of the CSRT tracker.
        Returns: (detected, cx_norm, cy_norm, distance_m, debug_frame)
        Updates self._tracker_failures on loss.
        """
        h, w = frame.shape[:2]
        debug = frame.copy()
        cx_norm = cy_norm = 0.5
        dist = 0.0

        success, (x, y, bw, bh) = self._tracker.update(frame)

        if not success or bw < 4 or bh < 4:
            self._tracker_failures += 1
            if self._tracker_failures >= TRACKER_FAIL_LIMIT:
                self.get_logger().warn(
                    f'Tracker lost target ({self._tracker_failures} failures) '
                    '— falling back to SAM3')
            return False, cx_norm, cy_norm, dist, debug

        self._tracker_failures = 0
        x, y, bw, bh = int(x), int(y), int(bw), int(bh)
        cx_px = x + bw / 2.0
        cy_px = y + bh / 2.0
        cx_norm = cx_px / w
        cy_norm = cy_px / h

        # Tracker runs every frame (no inference latency), so the latest
        # depth image already matches the RGB frame we're tracking on.
        # Tracker only has a bbox (no mask) — fall back to centroid-patch
        # depth sampling. SAM3 path uses the per-mask sampling above.
        dist_src = 'bbox'
        self._depth_for_sam3 = self._latest_depth   # alias for the helper
        d_depth = self._depth_distance_centroid_patch(cx_norm, cy_norm)
        if np.isfinite(d_depth) and d_depth > DEPTH_NEAR:
            dist = d_depth
            dist_src = 'depth'
            self._n_dist_depth += 1
        elif bh > 5:
            dist = (self.target_height * FX) / bh
            self._n_dist_bbox += 1

        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (255, 140, 0), 2)
        cv2.circle(debug, (int(cx_px), int(cy_px)), 7, (0, 255, 255), -1)
        label = f'[TRK:{dist_src}] {self.target_prompt}  {dist:.1f}m'
        cv2.putText(debug, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 140, 0), 2)

        return True, cx_norm, cy_norm, dist, debug

    # ──────────────────────────────────────────────────────────────────────
    # Publishing
    # ──────────────────────────────────────────────────────────────────────

    def _publish(self, header, detected, debug, cx, cy, dist, score):
        det_msg      = Bool();    det_msg.data   = self._detected
        cx_msg       = Float32(); cx_msg.data    = float(cx)
        cy_msg       = Float32(); cy_msg.data    = float(cy)
        dist_msg     = Float32(); dist_msg.data  = float(dist)
        score_msg    = Float32(); score_msg.data = float(score)

        self.pub_detected.publish(det_msg)
        self.pub_centroid_x.publish(cx_msg)
        self.pub_centroid_y.publish(cy_msg)
        self.pub_distance.publish(dist_msg)
        self.pub_score.publish(score_msg)

        # Encode debug image manually (same reason — avoid cv_bridge)
        debug_ros          = Image()
        debug_ros.header   = header
        debug_ros.height   = debug.shape[0]
        debug_ros.width    = debug.shape[1]
        debug_ros.encoding = 'bgr8'
        debug_ros.step     = debug.shape[1] * 3
        debug_ros.data     = debug.tobytes()
        self.pub_debug.publish(debug_ros)


# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SAM3DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
