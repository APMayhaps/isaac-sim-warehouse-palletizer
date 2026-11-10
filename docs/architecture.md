# Architecture and simulator notes

## Code layout

```
pallet_planner.py          pure Python: box sets, pallet optimizer, placement order
        │
palletizer_core.py         UR20 controller (one cell): IK, vacuum gripper, wrist weighing,
        │                  payload check, learning, results. Needs pxr (import after the app starts).
        ├── pallet_builder.py          front end: Isaac Sim standalone, one cell
        └── isaaclab_pallet_cells.py   front end: Isaac Lab, N cloned cells in the warehouse

warehouse_box_sorter.py    separate Franka small-box sorter (self-contained)
```

### The front-end interface

`palletizer_core.Palletizer` drives one cell. It's a generator: each `next()` is one physics step. It never touches simulator APIs directly; the front end hands it a robot and boxes that implement:

| Robot (`UR20Base`) | Box (`BoxBase`) |
|---|---|
| `q()` joint positions | `pose()` position and quaternion (w, x, y, z) |
| `flange_pose()` position and quaternion (w, x, y, z) | `lin_vel()` |
| `jacobian()` 6×6, flange link, world frame | `teleport(position, yaw)` |
| `set_targets(q)` | `spec`, `path`, `size`, `staging` |
| `flange_force()` wrist force/torque sensor reading | |
| `joint_load()` highest joint torque / its limit | |
| `configure()`, `flange_path` | |

`UR20Base` supplies `tcp_pose()` and the damped-least-squares `ik()` step on top of these.

* **`pallet_builder.py`** implements them with Isaac Sim's experimental core API (`Articulation`, `RigidPrim`).
* **`isaaclab_pallet_cells.py`** implements them on Isaac Lab's batched assets. `Cells` reads every cell's state once per physics step into numpy arrays: joint positions, flange pose, Jacobian, wrist force, joint torques and box poses. Each cell's `CellRobot` / `CellBox` then reads its own row. Joint targets from all cells are collected and written to Isaac Lab in one call.

**Cell origin.** Everything in the core is relative to a cell origin. The pick spot, ready pose, pallet, reject lane and the base-angle math all use it, so the same controller works for one cell at the world origin or many cells at Isaac Lab's environment origins.

### Per physics step (Isaac Lab)

```
for each cell:  next(controller)         # reads its row, sets its joint targets, may teleport a box / seal / release
cells.apply_targets()                    # all cells' joint targets -> robot.set_joint_position_target_index
scene.write_data_to_sim(); sim.step(); scene.update(dt)
cells.refresh()                          # new state for the next step
```

## Payload model (UR20, 20 kg)

| Constant (`palletizer_core.py`) | Value | Meaning |
|---|---|---|
| `PAYLOAD_RATED_KG` | 20.0 | UR20 rated payload at the flange |
| `TOOL_MASS` | 4.0 | vacuum gripper |
| `MAX_BOX_KG` | 16.0 | heaviest box: rated payload minus gripper |
| `WEIGH_TOLERANCE_KG` | 0.2 | a box weighed up to 16.2 kg is still accepted |

* **Tool mass.** The 4 kg is added to the `wrist_3_link` mass in both front ends, so joint torques and the wrist sensor include it.
* **Payload check.** After weighing, if box + gripper > 20 kg (+ tolerance), the arm sets the box back down and rejects it.
* **Torque limits.** The joint limits are the asset's (UR20 stock: 730 / 730 / 430 / 100 / 100 / 100 N·m). `joint_load()` reports the peak torque per box as a fraction of these, using the torques PhysX actually solved. In Isaac Lab, `applied_torque` for implicit servos is a clipped PD estimate that reads 100% whenever the arm moves, so the script uses `root_physx_view.get_dof_projected_joint_forces()` instead.
* **The planner** generates boxes up to `--max-box-mass` (default 16 kg), and every box set includes one box exactly that heavy.

## Suction model

* **Seal.** When the cups touch a box, a USD `PhysicsFixedJoint` is created between the flange link and the box, at their current relative pose. Release deletes it.
* **Why boxes never sleep.** A sleeping PhysX body ignores a joint added at runtime, and a velocity write doesn't wake it, so every box has a sleep threshold of 0.
* **Holding capacity.** Capacity = vacuum × cup area (78.5 cm²). Every step the load the cups must carry, `m · (a + g)`, is split into a pull part and a sideways part. The seal fails if `pull/capacity + sideways/(0.5 · capacity) > 1`, or if the box moves more than 8 mm relative to the cups.
* **Why not PhysX joint breaking.** PhysX's own joint break force can't be used. In a loop with the robot's articulation, the solver reports constraint forces many times the actual load, so seals broke under boxes of a few kg.
* **Regulation.** The vacuum is set per box from the measured weight and the planned peak acceleration (8–80 kPa).

## Weighing

The wrist force/torque sensor is the incoming joint force of `wrist_3_link`, which includes the gripper and anything sealed to it. It's zeroed (tared) once per episode with the gripper empty. Mass is `(|F_loaded| − |F_tare|) / g`, using medians of samples taken while the box is held still 8 cm up. A steadiness check compares the two halves of the samples.

## Simulator quirks worked around

* **The UR20 sits under a raised parent transform.** Its root is a fixed joint to the world at its starting pose. Moving the asset itself, or the joint's anchor, leaves the base at the floor, so the standalone script references it under a translated `Xform`. In Isaac Lab the articulation's `init_state.pos` does the same.
* **The gripper's look is a separate prim.** The UR20 asset's links are instance proxies, so no child prims can be added under the flange. The gripper visual is its own prim that follows the flange each frame.
* **Simulator readings are copied.** `.numpy()` of a simulator buffer can be a view that the next read overwrites, so values are copied.
* **Quaternion order.** Isaac Lab returns x, y, z, w; the core uses w, x, y, z. The Isaac Lab adapters convert.
* **The controller only advances while playing.** If it ran while the timeline was paused, its motion sequence would run ahead of the physics.
* **Stop clears seal joints.** Stop rewinds the robot and boxes but keeps prims added during the run. A leftover seal would yank the rewound robot and box together on the next Play, so Stop removes all seals.
* **The default joint pose must be written explicitly in Isaac Lab.** After `sim.reset()` the robot starts at its zero pose until `write_joint_position_to_sim_index` is called.
