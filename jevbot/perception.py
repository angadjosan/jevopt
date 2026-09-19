"""Turn a camera frame into facts a text-only model can read.

Jev accepts text only -- no images (https://docs.typesafe.ai/concepts/system-one).
So the pixels stop here: this module finds the apple in the RGB image, recovers
its 3D position from the depth buffer, and emits a small JSON observation. The
apple's true pose is never consulted; only the camera is.
"""

from __future__ import annotations

import numpy as np

MIN_BLOB_PIXELS = 12


def find_apple_pixels(rgb: np.ndarray) -> np.ndarray:
    """Mask of 'red enough to be the apple' pixels.

    The apple is the only saturated red thing in the scene; the table is wood
    brown and the robot is white, so a channel-dominance test is enough.
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r > 90) & (r - g > 55) & (r - b > 55)


APPLE_RADIUS = 0.026


def camera_eye(view) -> np.ndarray:
    """Recover the camera position from its view matrix."""
    view_m = np.array(view, dtype=np.float64).reshape(4, 4).T
    return (-view_m[:3, :3].T @ view_m[:3, 3])


def deproject(u: int, v: int, depth_buffer: float, view, proj, width: int, height: int):
    """Pixel + depth -> world coordinates."""
    view_m = np.array(view, dtype=np.float64).reshape(4, 4).T
    proj_m = np.array(proj, dtype=np.float64).reshape(4, 4).T
    inv = np.linalg.inv(proj_m @ view_m)
    ndc = np.array([
        (2.0 * u - width) / width,
        -(2.0 * v - height) / height,
        2.0 * depth_buffer - 1.0,
        1.0,
    ])
    world = inv @ ndc
    return (world[:3] / world[3]).tolist()


def observe(rgb, depth, view, proj) -> dict:
    """Everything the policy is allowed to know about the apple."""
    height, width = depth.shape
    mask = find_apple_pixels(rgb)
    count = int(mask.sum())
    if count < MIN_BLOB_PIXELS:
        return {"apple_visible": False, "apple_pixels": count}

    vs, us = np.nonzero(mask)
    u = int(round(us.mean()))
    v = int(round(vs.mean()))
    # Median depth over the blob: robust to the rim pixels that straddle the
    # silhouette and read the table behind the apple.
    d = float(np.median(depth[mask]))
    surface = np.array(deproject(u, v, d, view, proj, width, height))
    # The depth hit lies on the apple's near surface. The centre is one radius
    # further along the camera ray.
    ray = surface - camera_eye(view)
    centre = surface + APPLE_RADIUS * ray / np.linalg.norm(ray)
    x, y, z = centre

    return {
        "apple_visible": True,
        "apple_pixels": count,
        "apple_image_uv": [u, v],
        "apple_image_frac": [round(u / width, 3), round(v / height, 3)],
        "apple_world_est": [round(x, 3), round(y, 3), round(z, 3)],
    }


# Bucket edges, in metres. Jev "struggles with tasks that require numeric
# precision" and is "not a calculator" -- the vendor's guidance is to keep the
# arithmetic in code and hand the model semantic facts. So every threshold in
# this file lives here, and the model sees words.
ALIGNED = 0.02        # horizontal offset we call "lined up"
NEAR = 0.07           # below this an offset is "slightly", above it "far"
GRIP_HEIGHT = 0.025   # vertical gap at which the fingers straddle the apple
LIFTED_CLEAR = 0.10   # height above the table that counts as picked up


def _axis(offset: float, positive: str, negative: str) -> str:
    """Turn a signed offset into a phrase."""
    if abs(offset) < ALIGNED:
        return "lined up"
    where = positive if offset > 0 else negative
    return f"{'slightly' if abs(offset) < NEAR else 'far'} {where}"


def describe(obs: dict, ee, fingers: str, holding: bool,
             step: int, last_action: str | None, last_result: str | None,
             table_top: float) -> dict:
    """Assemble the state handed to Jev.

    Deliberately lean and worded, not numeric: a long state "full of irrelevant
    detail" is a documented failure mode too.
    """
    state = {
        "step": step,
        "fingers": fingers,
        "fingers_are_gripping_an_object": "yes" if holding else "no",
    }
    if last_action:
        state["last_action"] = last_action
        state["last_action_result"] = last_result

    if not obs.get("apple_visible"):
        state["apple_seen_by_camera"] = "no"
        state["where_the_apple_is"] = "unknown -- the camera cannot see it"
        return state

    ax, ay, az = obs["apple_world_est"]
    dx, dy, dz = ax - ee[0], ay - ee[1], az - ee[2]
    horizontal = float(np.hypot(dx, dy))
    over = horizontal < ALIGNED
    at_height = abs(dz) < GRIP_HEIGHT

    state["apple_seen_by_camera"] = "yes"
    state["where_the_apple_is_relative_to_the_gripper"] = {
        "along_forward_back_axis": _axis(dx, "forward of the gripper",
                                         "behind the gripper"),
        "along_left_right_axis": _axis(dy, "to the left of the gripper",
                                       "to the right of the gripper"),
        "height": ("level with the gripper" if at_height else
                   "below the gripper" if dz < 0 else "above the gripper"),
    }
    state["gripper_is_horizontally_over_the_apple"] = "yes" if over else "no"
    state["gripper_is_at_the_right_height_to_grip"] = "yes" if at_height else "no"
    state["gripper_is_held_high_above_the_table"] = (
        "yes" if ee[2] - table_top > LIFTED_CLEAR else "no")
    return state
