import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("./model.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

HAND_INIT_POS = np.array((0, 0.6, 0.2))
HAND_INIT_QUAT = np.array([1, 0, 1, 0])
CTRL = np.array([-1, 1])
MOCAP_ID = model.body_mocapid[data.body("mocap").id]

data.mocap_pos[MOCAP_ID][:] = HAND_INIT_POS
data.mocap_quat[MOCAP_ID][:] = HAND_INIT_QUAT
data.ctrl[:] = CTRL

print("Original state:")
print("- mocap pos:", data.mocap_pos[MOCAP_ID][:])
print("- mocap quat:", data.mocap_quat[MOCAP_ID][:])
print("- qacc:", data.qacc[:])
print("- qvel:", data.qvel[:])
print("- qpos:", data.qpos[:])
print("- ctrl:", data.ctrl[:])

# simulate
mujoco.mj_step(model, data)
mujoco.mj_rnePostConstraint(model, data)

print("\nAfter 1 simulation step:")
print("- mocap pos:", data.mocap_pos[MOCAP_ID][:])
print("- mocap quat:", data.mocap_quat[MOCAP_ID][:])
print("- qacc:", data.qacc[:])
print("- qvel:", data.qvel[:])
print("- qpos:", data.qpos[:])
print("- ctrl:", data.ctrl[:])
