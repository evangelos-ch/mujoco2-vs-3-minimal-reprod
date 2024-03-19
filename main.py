import mujoco  # type: ignore
import numpy as np

np.set_printoptions(suppress=True)

model = mujoco.MjModel.from_xml_path("./model.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

HAND_INIT_POS = np.array((0, 0.6, 0.2))
HAND_INIT_QUAT = np.array([1, 0, 1, 0])
MOCAP_ID = model.body_mocapid[data.body("mocap").id]

data.mocap_pos[MOCAP_ID][:] = HAND_INIT_POS
data.mocap_quat[MOCAP_ID][:] = HAND_INIT_QUAT

# simulate
mujoco.mj_step(model, data)

print("After 1 step:")
print("- qacc:", data.qacc[:])
print("- qvel:", data.qvel[:])
print("- qpos:", data.qpos[:])
print("- time", data.time)
