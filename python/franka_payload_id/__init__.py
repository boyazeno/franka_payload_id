"""Payload inertial-parameter identification for the Franka Emika Panda.

The pipeline identifies the ten inertial parameters of a tool bolted to the flange
(``panda_link8``) so they can be entered into Desk's end-effector parameter fields:

    mass m [kg], centre of mass w.r.t. the flange [m], inertia about the CoM [kg m^2]

See ``docs/THEORY.md`` for the derivations and a formula-to-code map.
"""

__version__ = "0.1.0"

FLANGE_FRAME = "panda_link8"
"""Franka's flange frame F. Verified equal to the ``panda_link7`` frame translated by
(0, 0, 0.107) with identity rotation, from both the FCI DH table and the URDF."""

N_JOINTS = 7
N_PARAMS = 10
"""Inertial parameters per rigid body: [m, m*cx, m*cy, m*cz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]."""
