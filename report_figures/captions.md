# Section 4 figures: captions

Each figure has a PNG (300 dpi, for the report) and an SVG (editable text, opens in Figma, draw.io, Inkscape or Word). They were drawn with Python and matplotlib (`make_figures.py`); edit a label there and re-run it to regenerate all six.

---

**Figure 4.1: System architecture** (`fig4-1_system_architecture`)

The cell has seven parts: the infeed, vision, safety, the cell controller's five software modules, the robot, the vacuum system and the end of arm. The colored lines show how they connect: data and commands, sensor feedback, compressed air and vacuum, safety signals, and mechanical links. The checkweigher's weight feeds both the planner and the supervisor, and the wrist sensor feeds the measured weight back, so weight links planning, gripping and motion. The safety controller can stop the robot at any time, and the check valve holds vacuum during a stop so a box isn't dropped.

**Figure 4.2: Per-box control flowchart** (`fig4-2_per_box_flowchart`)

This is the decision sequence the supervisor runs for every box, from release at the buffer to logging the result. Each check has a defined fault response (orange):
- vision can't find the box,
- the seal isn't confirmed,
- the weight doesn't match the checkweigher,
- the box is over the payload,
- vacuum is lost mid-carry,
- the placement is off.

Retries and timeouts keep one bad box from stopping the line, and anything that could drop a box triggers a protective stop.

**Figure 4.3: One pick-and-place cycle** (`fig4-3_cycle_swimlane`)

The swimlane shows which subsystem acts at each moment of one cycle, left to right in time: feed, pick, weigh and set grip, carry and place, verify. Handoffs between lanes are the interfaces described in the System Overview. Vision locates the box before the pick and measures it again before the release, and the vacuum is monitored for the whole carry.

**Figure 4.4: Vacuum circuit and holding capacity** (`fig4-4_vacuum_circuit`)

Compressed air drives a venturi vacuum generator. A proportional regulator then sets the vacuum between 8 and 80 kPa from each box's weight, and a pressure sensor confirms the seal. A check valve holds vacuum if air is lost, and a blow-off pulse releases the box cleanly. The calculation panel shows the 628 N full-vacuum capacity of the four 50 mm cups, and the vacuum the simulation actually used for 3.3, 7.3 and 16 kg boxes (well below full vacuum, which saves compressed air).

**Figure 4.5: Weight data flow** (`fig4-5_weight_data_flow`)

This figure shows the design's central idea: one measured quantity, the box's weight, drives all three decisions.
- **Planning:** the checkweigher's listed weight plans the pallet before the first pick.
- **Timing and grip:** the wrist sensor's measured weight, cross-checked against the listed one, sets the move timing and the vacuum.
- **Coupling:** the planned spot sets the path length, and the move timing sets the peak acceleration the grip must cover, so the three decisions can't contradict each other.

**Figure 4.6: Cell layout, top and side view** (`fig4-6_cell_layout`)

The top view is to scale, using the simulated cell's dimensions:
- the UR20 on its pedestal at the center, with its 1.75 m reach,
- the 1.0 × 1.0 m pallet 0.88 m in front of it,
- the infeed line (dimensioner, checkweigher, sequencing buffer) feeding the pick stop,
- both camera views, and a proposed 4 × 4 m fence with an interlocked pallet-exit gate.

The side view shows why the 0.7 m pedestal and the 1.25 m transit height work. The arm reaches the bottom layer, yet a carried 26 cm box still clears the tallest possible 3-layer stack (0.92 m) by at least 6 cm.

---

## Notes for the report

* **Proposed versus shown in simulation.** The simulation (GitHub repo) demonstrates the planner, weight measurement, weight-matched vacuum, payload check, motion timing, placement correction and the stability check.
  * **Proposed for the real cell (not in the simulation):** the cameras and vision software (in simulation, box positions come straight from the physics engine), the vacuum generator, regulator, check valve, blow-off valve and pressure sensor (the simulation models the seal as a force limit), the safety controller, fence, gate and E-stops, and the fault responses for vision misses and seal confirmation.
  * Say so in the report if a judge asks what is proven.
* **Thresholds chosen for the design:** the 0.5 s seal-confirmation timeout, the 5 mm placement tolerance, 2 retries and the 6 bar air supply are typical values, not measured ones. The 5 % weight check and the 2–5 mm accuracy target match the Problem Statement.
* **Inconsistency in Section 3.** The System Overview says boxes of "1.5 to 15 kg" and "a 15 kg box plus the tool stays under the 20 kg payload". The rest of the report, and these figures, use 16 kg (16 kg + 4 kg tool = the full 20 kg). Update Section 3 to 16 kg to match.
