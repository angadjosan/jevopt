"""A Franka Panda on a table with an apple, driven by discrete primitives.

The primitives in ACTIONS are the only way to move the robot -- they are the
"tool calls" that Jev chooses between. Ground-truth object state lives here but
is deliberately kept out of what the policy sees: the policy is handed only
proprioception (joint-derived gripper pose) and whatever perception.py can
recover from the camera image.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pybullet as p
import pybullet_data

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

TABLE_TOP = 0.626
ARM_JOINTS = list(range(7))
FINGERS = (9, 10)
EE_LINK = 11            # panda_grasptarget
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0
XY_STEP = 0.03
Z_STEP = 0.025
CAMERA_EYE = (1.35, -0.95, 1.25)
START_POSE = (0.15, 0.0, TABLE_TOP + 0.18)

# The action space. Every entry becomes an option in Jev's `choice` question,
# so the text here is what the model actually reads -- keep it behavioural.
ACTIONS: dict[str, str] = {
    "move_forward": "Move the gripper 3cm forward. Pick this when the apple is "
                    "forward of the gripper and they are not lined up yet.",
    "move_back": "Move the gripper 3cm back. Pick this when the apple is behind "
                 "the gripper and they are not lined up yet.",
    "move_left": "Move the gripper 3cm to the left. Pick this when the apple is "
                 "to the left of the gripper and they are not lined up yet.",
    "move_right": "Move the gripper 3cm to the right. Pick this when the apple "
                  "is to the right of the gripper and they are not lined up yet.",
    "descend": "Lower the gripper 2.5cm toward the table. Pick this when the "
               "fingers are open and the gripper is already horizontally over "
               "the apple but is still too high to grip it.",
    "ascend": "Raise the gripper 2.5cm away from the table. Pick this to lift "
              "the apple once the fingers are gripping it, or to back off if "
              "the gripper is too low.",
    "open_gripper": "Open the fingers. Pick this when the fingers are closed "
                    "but are not gripping anything, so they are ready to try "
                    "again, or to deliberately release an object.",
    "close_gripper": "Close the fingers to grip the apple. Pick this when the "
                     "gripper is horizontally over the apple AND at the right "
                     "height to grip it AND the fingers are still open.",
    "done": "Stop: the task is finished. Pick this only when the fingers are "
            "already gripping the apple AND the gripper is held high above the "
            "table.",
}

# Workspace limits, so a bad action cannot fling the arm out of reach.
X_RANGE = (-0.10, 0.50)
Y_RANGE = (-0.40, 0.40)
Z_RANGE = (TABLE_TOP + 0.015, TABLE_TOP + 0.45)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ArmWorld:
    """Panda + table + apple, stepped through discrete primitives."""

    def __init__(self, apple_xy: tuple[float, float] = (0.45, 0.10), seed: int = 0):
        self.apple_xy = apple_xy
        self.rng = np.random.default_rng(seed)
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self) -> None:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane.urdf")
        self.table = p.loadURDF("table/table.urdf", [0.25, 0, 0], useFixedBase=True)
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf", [-0.35, 0, TABLE_TOP], useFixedBase=True
        )
        self.apple = p.loadURDF(
            os.path.join(ASSETS, "apple.urdf"),
            [self.apple_xy[0], self.apple_xy[1], TABLE_TOP + 0.027],
        )
        # An apple is not a billiard ball: damp the rolling so it stays put.
        p.changeDynamics(self.apple, -1, lateralFriction=1.4, rollingFriction=0.004,
                         spinningFriction=0.004, restitution=0.0)
        p.changeDynamics(self.table, -1, lateralFriction=1.0)
        for finger in FINGERS:
            p.changeDynamics(self.robot, finger, lateralFriction=2.0,
                             rollingFriction=0.001, spinningFriction=0.001)

        # A neutral, elbow-up pose with the hand above the table.
        self.rest = [0.0, -0.45, 0.0, -2.0, 0.0, 1.6, 0.785]
        for i, angle in zip(ARM_JOINTS, self.rest):
            p.resetJointState(self.robot, i, angle)
        for finger in FINGERS:
            p.resetJointState(self.robot, finger, FINGER_OPEN)

        self.finger_target = FINGER_OPEN
        self.grasp_orn = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
        self.steps = 0
        self._settle(60)
        # Start from a consistent pose above the table, not wherever the rest
        # joint angles happen to put the hand.
        self.target = list(START_POSE)
        self._goto(self.target, 260)

    # ------------------------------------------------------------------ state

    def ee_pos(self) -> tuple[float, float, float]:
        """Gripper position from forward kinematics -- proprioception."""
        return tuple(p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=1)[0])

    def finger_gap(self) -> float:
        return sum(p.getJointState(self.robot, f)[0] for f in FINGERS)

    def fingers_commanded(self) -> str:
        return "closed" if self.finger_target < 0.02 else "open"

    def object_between_fingers(self) -> bool:
        """Commanded shut but stalled open -- something is in the way.

        Pure joint sensing: it says something is held, not what. A real gripper
        knows exactly this much.
        """
        return self.finger_target < 0.02 and self.finger_gap() > 0.012

    def apple_pos(self) -> tuple[float, float, float]:
        """Ground truth. Scoring only -- never handed to the policy."""
        return tuple(p.getBasePositionAndOrientation(self.apple)[0])

    def grasped(self) -> bool:
        """True when both fingers touch the apple and it is off the table."""
        touching = sum(
            1 for f in FINGERS
            if p.getContactPoints(bodyA=self.robot, bodyB=self.apple, linkIndexA=f)
        )
        return touching == 2 and self.apple_pos()[2] > TABLE_TOP + 0.045

    def apple_lifted(self, clearance: float = 0.08) -> bool:
        return self.grasped() and self.apple_pos()[2] > TABLE_TOP + clearance

    # ---------------------------------------------------------------- control

    def _settle(self, steps: int) -> None:
        for _ in range(steps):
            for finger in FINGERS:
                p.setJointMotorControl2(self.robot, finger, p.POSITION_CONTROL,
                                        targetPosition=self.finger_target, force=70)
            p.stepSimulation()

    def _goto(self, target: list[float], steps: int = 110) -> None:
        angles = p.calculateInverseKinematics(
            self.robot, EE_LINK, target, self.grasp_orn,
            restPoses=self.rest + [FINGER_OPEN, FINGER_OPEN],
            maxNumIterations=200, residualThreshold=1e-4,
        )
        for i in ARM_JOINTS:
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL,
                                    targetPosition=angles[i], force=240,
                                    maxVelocity=1.6)
        self._settle(steps)

    def apply(self, action: str) -> str:
        """Run one primitive. Returns a short note about what happened."""
        if action not in ACTIONS:
            return f"unknown action {action!r}, ignored"
        self.steps += 1

        if action == "done":
            return "stopped"
        if action == "open_gripper":
            self.finger_target = FINGER_OPEN
            self._settle(90)
            return "fingers opened"
        if action == "close_gripper":
            self.finger_target = FINGER_CLOSED
            self._settle(130)
            return "fingers closed"

        delta = {
            "move_forward": (XY_STEP, 0, 0), "move_back": (-XY_STEP, 0, 0),
            "move_left": (0, XY_STEP, 0), "move_right": (0, -XY_STEP, 0),
            "descend": (0, 0, -Z_STEP), "ascend": (0, 0, Z_STEP),
        }[action]

        # Track a commanded target rather than the measured pose, so repeated
        # small moves do not accumulate the IK/servo error.
        wanted = [self.target[i] + delta[i] for i in range(3)]
        self.target = [
            _clamp(wanted[0], *X_RANGE),
            _clamp(wanted[1], *Y_RANGE),
            _clamp(wanted[2], *Z_RANGE),
        ]
        self._goto(self.target)

        # If IK could not follow, snap the command back to where the arm
        # actually is. Otherwise the commanded target drifts away from reality
        # and every later move is computed from a pose the arm never reached.
        reached = self.ee_pos()
        drift = max(abs(self.target[i] - reached[i]) for i in range(3))
        if drift > 0.04:
            self.target = list(reached)
            return "the arm could not move there -- it is at its reach limit"
        if self.target != wanted:
            return "moved, but hit the edge of the workspace"
        return "moved"

    # ----------------------------------------------------------------- camera

    def camera(self, width: int = 224, height: int = 224):
        """Render the scene from a fixed over-the-shoulder camera."""
        view = p.computeViewMatrix(
            cameraEyePosition=list(CAMERA_EYE),
            cameraTargetPosition=[0.15, 0.0, 0.72],
            cameraUpVector=[0, 0, 1],
        )
        near, far = 0.05, 3.5
        proj = p.computeProjectionMatrixFOV(
            fov=50.0, aspect=width / height, nearVal=near, farVal=far
        )
        _, _, rgb, depth, _ = p.getCameraImage(
            width, height, view, proj, renderer=p.ER_TINY_RENDERER
        )
        rgb = np.reshape(np.array(rgb, dtype=np.uint8), (height, width, 4))[:, :, :3]
        depth = np.reshape(np.array(depth, dtype=np.float32), (height, width))
        return rgb, depth, view, proj, (near, far)

    def close(self) -> None:
        p.disconnect(self.client)
