"""Real-robot support for the Franka cell driven by ``franky_service``.

TiPToP already targets a real Franka, but through ``bamboo-franka-controller``
(ZMQ :5555/:5559) plus a ZED camera. Our cell is driven by ``franky_service``,
which speaks the OpenPI WebSocket policy protocol instead: it connects to a
policy server, streams observations, and expects action chunks back.

This package adapts TiPToP to that cell WITHOUT touching any shared TiPToP
module, so the sim/robolab path and the bamboo path keep working:

``frame_to_h5``   driver observation -> the H5 snapshot ``tiptop_h5.py`` already
                  consumes. No ZED and no FoundationStereo: the driver supplies
                  metric depth directly, matching the other baselines and our
                  own method.
``franky_client`` a third ``RobotClient`` backend beside ``BambooFrankaClient``
                  and ``UR5Client``, so ``execute_plan.execute_cutamp_plan``
                  can drive the arm over the same WebSocket.

The driver never waits: it polls for an action chunk at a fixed rate. Any
blocking call inside the request handler would deadlock it, so the client keeps
a trajectory cursor and always answers immediately (the pattern proven in the
cap-x adapter).
"""

from tiptop.franky.frame_to_h5 import (
    DRIVER_DEPTH_KEYS,
    DRIVER_RGB_KEYS,
    observation_to_h5,
    wire_to_observation_arrays,
)

__all__ = [
    "DRIVER_DEPTH_KEYS",
    "DRIVER_RGB_KEYS",
    "observation_to_h5",
    "wire_to_observation_arrays",
]
