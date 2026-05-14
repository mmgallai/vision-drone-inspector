#!/usr/bin/env python3
"""
Scene Randomizer
================
Shuffles which physical object lives at each of the five canonical
target positions in scene1_hydrant.

The mission node has zero advance knowledge of object positions
(`expected_target_xy = None` everywhere) — it identifies whatever it
visually sees. So if we permute the objects among the five canonical
slots before each run, every mission becomes a real test of the
SAM3+triangulation autonomy.

Canonical slot positions (from the SDF defaults):
    A: (0.0,  0.0)   — originally fire hydrant
    B: (-3.5,  1.5)  — originally potted plant
    C: (-3.5, -1.5)  — originally park bench
    D: (1.5, -2.0)   — originally trash bin
    E: (1.5,  2.0)   — originally mailbox

How it works:
  1. Wait for the bridged `/world/<world>/set_pose` service to come up.
  2. Generate a random permutation of the 5 object names mapping to
     the 5 slots (optionally with a fixed `seed` ROS parameter for
     reproducible runs).
  3. Issue one SetEntityPose call per object via async client, then
     spin until they all complete (or a 10 s timeout fires).
  4. Log the resulting permutation in human-readable form so the test
     output records what the drone is actually facing.

Disable with `randomize:=false` in the launch — this node simply isn't
included in that case.
"""
import math
import random
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose


WORLD_NAME = 'recon_world'

# Object name in SDF  →  canonical (x, y) "slot" the SDF places it at.
# The fire hydrant is wrapped in an `<include>` with `<name>target</name>`,
# so its model name is "target". The other four are direct `<model name=…>`
# entries.
SLOTS = {
    'target':       (0.0,  0.0),    # fire hydrant
    'potted_plant': (-3.5,  1.5),
    'park_bench':   (-3.5, -1.5),
    'trash_bin':    (1.5, -2.0),
    'mailbox':      (1.5,  2.0),
}

# Pretty names for the log line so the human reading it can map the SDF
# model name to the SAM3 prompt that targets that object.
PROMPT = {
    'target':       'fire hydrant',
    'potted_plant': 'potted plant',
    'park_bench':   'bench',
    'trash_bin':    'trash bin',
    'mailbox':      'mailbox',
}


class SceneRandomizer(Node):
    def __init__(self):
        super().__init__('scene_randomizer')

        # Optional reproducibility — `seed:=N` in the launch (or
        # `--ros-args -p seed:=42`). 0 means "use OS entropy".
        self.declare_parameter('seed', 0)
        seed_val = int(self.get_parameter('seed').value)
        self._rng = random.Random(seed_val) if seed_val else random.Random()

        self._cli = self.create_client(
            SetEntityPose, f'/world/{WORLD_NAME}/set_pose')

        # Hold all in-flight futures so we can wait on them collectively
        self._pending: list = []

    def wait_for_service(self, timeout_s: float = 30.0) -> bool:
        """Block until the bridged /world/<world>/set_pose service is up,
        or the timeout fires. Returns True iff the service became ready."""
        log = self.get_logger()
        log.info(f'Waiting for /world/{WORLD_NAME}/set_pose '
                 f'(up to {timeout_s:.0f}s) ...')
        deadline = time.monotonic() + timeout_s
        while not self._cli.service_is_ready():
            if time.monotonic() > deadline:
                log.error(
                    f'/world/{WORLD_NAME}/set_pose did not appear within '
                    f'{timeout_s:.0f}s — bridge not running? Skipping '
                    'randomization; objects stay at SDF defaults.')
                return False
            rclpy.spin_once(self, timeout_sec=0.5)
        return True

    def shuffle_and_send(self) -> Optional[dict]:
        """Build a random object↔slot mapping and dispatch one
        SetEntityPose request per object. Returns the mapping dict
        (`object_name -> (x, y)`) on success, None if the service was
        never ready."""
        if not self.wait_for_service():
            return None

        objects = list(SLOTS.keys())
        slots   = list(SLOTS.values())
        self._rng.shuffle(slots)
        mapping = dict(zip(objects, slots))

        for name, (x, y) in mapping.items():
            req = SetEntityPose.Request()
            req.entity.name = name
            req.entity.type = 2   # ros_gz_interfaces/Entity.MODEL
            req.pose.position.x = float(x)
            req.pose.position.y = float(y)
            req.pose.position.z = 0.0
            # All targets keep yaw=0 (matches their SDF defaults).
            req.pose.orientation.w = 1.0
            self._pending.append(self._cli.call_async(req))

        # Spin until all calls return or 10 s elapses.
        deadline = time.monotonic() + 10.0
        while any(not f.done() for f in self._pending):
            if time.monotonic() > deadline:
                self.get_logger().warn(
                    'Some SetEntityPose calls did not return within 10s; '
                    'objects may be partially randomized.')
                break
            rclpy.spin_once(self, timeout_sec=0.2)

        return mapping

    def report(self, mapping: dict) -> None:
        log = self.get_logger()
        log.info(' Scene randomized — object → slot:')
        for name, (x, y) in mapping.items():
            origin = SLOTS[name]
            if (x, y) == origin:
                tag = 'UNCHANGED'
            else:
                # Whichever object's canonical slot this NEW position
                # was — that object got displaced and now lives elsewhere.
                original_owner = next(k for k, v in SLOTS.items() if v == (x, y))
                tag = f"now in {PROMPT[original_owner]}'s old slot"
            log.info(
                f'   {PROMPT[name]:14s} ({name:13s}) → '
                f'({x:+.2f}, {y:+.2f})   [{tag}]')


def main(args=None):
    rclpy.init(args=args)
    node = SceneRandomizer()
    try:
        mapping = node.shuffle_and_send()
        if mapping is not None:
            node.report(mapping)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
