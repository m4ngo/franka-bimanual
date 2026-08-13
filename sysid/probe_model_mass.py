"""Read-only libfranka Model.mass() probe. NO motion, no torque control.

RobotState.q is writable and Model.mass(state) forwards to libfranka's q-overload,
so the model can be evaluated at ANY configuration without the arm going there.
Runs on the NUC, where pylibfranka lives.
"""
import json
import sys

import numpy as np
import pylibfranka

ROBOT_IP = sys.argv[1]
POSES = json.loads(sys.argv[2])          # {name: [q1..q7]}

robot = pylibfranka.Robot(ROBOT_IP, pylibfranka.RealtimeConfig.kIgnore)
model = robot.load_model()
state = robot.read_once()                # one read, to get a RobotState to mutate

out = {}
for name, q in POSES.items():
    state.q = tuple(float(x) for x in q)
    M = np.asarray(model.mass(state), dtype=np.float64).reshape(7, 7, order="F")
    out[name] = dict(q=list(q), diag=np.diag(M).tolist())

print("PROBE_JSON " + json.dumps(out))
