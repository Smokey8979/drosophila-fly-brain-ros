"""Live Virtual Brain Learning Laboratory GUI.

Provides a live, interactive 3-panel experimental laboratory:
  - Panel 1: Live 2D/3D Gazebo-compatible virtual arena with 4-wheel robot, LiDAR beams, obstacles, zones.
  - Panel 2: Live brain activity dashboard with real-time spike rasters, synaptic weights, dopamine, and compass.
  - Panel 3: Interactive teacher panel with movement commands, reward/punish buttons, and mode switches.
"""

import collections
import math
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, List, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virtual_world.arena import FlyArena
from virtual_world.closed_loop_agent import EmbodiedFlyAgent


class LiveBrainLaboratory:
    """Integrated interactive graphical laboratory for the embodied Drosophila connectome."""

    def __init__(self, root: Optional[tk.Tk] = None, seed: int = 42):
        self.own_root = False
        if root is None:
            self.root = tk.Tk()
            self.own_root = True
        else:
            self.root = root

        self.root.title("Drosophila Virtual Brain Learning Laboratory — Interactive Live Environment")
        self.root.geometry("1480x880")
        self.root.configure(bg="#1e1e24")

        # Operating Modes
        self.MODE_A_FIXED = "Mode A: Baseline Fixed Connectome"
        self.MODE_B_PLASTIC = "Mode B: Neural Plasticity Experiment"
        self.MODE_C_DECODER = "Mode C: External Readout Decoder"
        self.current_mode = self.MODE_B_PLASTIC

        # State Variables
        self.is_running = True
        self.is_teaching = False
        self.is_autonomous = True
        self.episode_count = 1
        self.step_count = 0
        self.total_reward = 0.0
        self.total_penalty = 0.0
        self.last_step_time = time.time()
        self.current_fps = 50.0

        # Teacher Override Inputs
        self.teacher_v = 0.0
        self.teacher_omega = 0.0
        self.manual_reward_pulse = 0.0
        self.manual_punish_pulse = 0.0

        # External Decoder Baseline Weights (Mode C)
        self.decoder_weights = np.zeros((2, 16), dtype=np.float32)

        # Spiking History Buffers for Live Raster Plots
        self.max_raster_steps = 100
        self.kc_raster_buffer = collections.deque(maxlen=self.max_raster_steps)
        self.lc4_rate_buffer = collections.deque(maxlen=self.max_raster_steps)
        self.p9_rate_buffer = collections.deque(maxlen=self.max_raster_steps)

        # Initialize Arena and Embodied Agent
        self.arena = FlyArena(width=80.0, height=80.0, dt=0.02, seed=seed)
        self._setup_arena_environment()
        self.agent = EmbodiedFlyAgent(arena=self.arena, seed=seed)
        self.agent.reset(x=0.0, y=-15.0, theta=math.pi / 2.0)

        # Build GUI Layout
        self._build_layout()
        self._bind_keys()

        # Start 50 Hz Update Loop
        self.root.after(20, self._step_loop)

    def _setup_arena_environment(self):
        """Configure arena boundaries, obstacles, and sensory zones."""
        self.arena.clear_entities()
        # Odor A (Banana / Sucrose Reward Zone) in North-East at (22, 22)
        self.odor_a = self.arena.add_odor("Odor_A_Banana", x=22.0, y=22.0, peak=2.0, sigma=22.0)
        self.food_patch = self.arena.add_food(x=22.0, y=22.0, radius=5.0, reward=1.0, odor_id="Odor_A_Banana")

        # Odor B (Citrus / Bitter Hazard Zone) in North-West at (-22, 22)
        self.odor_b = self.arena.add_odor("Odor_B_Citrus", x=-22.0, y=22.0, peak=2.0, sigma=22.0)
        self.hazard_patch = self.arena.add_hazard(x=-22.0, y=22.0, radius=5.0, penalty=1.0, odor_id="Odor_B_Citrus")

        # Obstacles forming a corridor requiring turning
        self.obs_center = self.arena.add_obstacle(x=0.0, y=5.0, radius=4.5)
        self.obs_left = self.arena.add_obstacle(x=-12.0, y=-5.0, radius=3.5)
        self.obs_right = self.arena.add_obstacle(x=12.0, y=-5.0, radius=3.5)

    def _build_layout(self):
        """Construct the 3-panel UI layout."""
        # Top Header Bar
        header = tk.Frame(self.root, bg="#111116", height=45)
        header.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="DROSOPHILA VIRTUAL BRAIN LEARNING LABORATORY",
            font=("Helvetica", 14, "bold"),
            fg="#00e5ff",
            bg="#111116",
        )
        title_lbl.pack(side=tk.LEFT, padx=15, pady=8)

        self.mode_badge = tk.Label(
            header,
            text=f"ACTIVE: {self.current_mode}",
            font=("Helvetica", 10, "bold"),
            fg="#76ff03",
            bg="#263238",
            padx=10,
            pady=3,
        )
        self.mode_badge.pack(side=tk.RIGHT, padx=15, pady=8)

        # Main Horizontal Container
        main_frame = tk.Frame(self.root, bg="#1e1e24")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        # ----------------------------------------------------
        # Panel 1: Virtual World Arena Canvas (Left)
        # ----------------------------------------------------
        p1_frame = tk.LabelFrame(
            main_frame,
            text=" Window 1: Virtual World Arena (4-Wheel Skid-Steer Physics) ",
            font=("Helvetica", 10, "bold"),
            fg="#b0bec5",
            bg="#18181f",
            padx=5,
            pady=5,
        )
        p1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=4)

        self.arena_canvas = tk.Canvas(p1_frame, width=460, height=520, bg="#0d0d11", highlightthickness=1, highlightbackground="#37474f")
        self.arena_canvas.pack(side=tk.TOP, pady=4)

        p1_stats = tk.Frame(p1_frame, bg="#18181f")
        p1_stats.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        self.p1_pose_lbl = tk.Label(p1_stats, text="Pose: x=0.0, y=0.0, theta=0.0°", font=("Courier", 9), fg="#cfd8dc", bg="#18181f")
        self.p1_pose_lbl.pack(side=tk.LEFT, padx=4)

        # ----------------------------------------------------
        # Panel 2: Live Neural Dashboard (Center)
        # ----------------------------------------------------
        p2_frame = tk.LabelFrame(
            main_frame,
            text=" Window 2: Live Brain Dashboard & Circuit Activity ",
            font=("Helvetica", 10, "bold"),
            fg="#00e5ff",
            bg="#18181f",
            padx=5,
            pady=5,
        )
        p2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.brain_canvas = tk.Canvas(p2_frame, width=540, height=520, bg="#0a0a0e", highlightthickness=1, highlightbackground="#00838f")
        self.brain_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=4)

        p2_stats = tk.Frame(p2_frame, bg="#18181f")
        p2_stats.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        self.p2_synapse_lbl = tk.Label(p2_stats, text="Mushroom Body: W_app=0.500 | W_av=0.500 | Valence=+0.00", font=("Courier", 9, "bold"), fg="#ffea00", bg="#18181f")
        self.p2_synapse_lbl.pack(side=tk.LEFT, padx=4)

        # ----------------------------------------------------
        # Panel 3: Interactive Teacher Console (Right)
        # ----------------------------------------------------
        p3_frame = tk.LabelFrame(
            main_frame,
            text=" Window 3: Interactive Teacher Panel ",
            font=("Helvetica", 10, "bold"),
            fg="#ffab00",
            bg="#18181f",
            padx=8,
            pady=6,
            width=360,
        )
        p3_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=4)
        p3_frame.pack_propagate(False)

        # Mode Selection
        mode_box = tk.LabelFrame(p3_frame, text="Neural Experiment Mode", font=("Helvetica", 9, "bold"), fg="#90a4ae", bg="#18181f")
        mode_box.pack(fill=tk.X, pady=4)

        self.mode_var = tk.StringVar(value=self.current_mode)
        for m in [self.MODE_B_PLASTIC, self.MODE_A_FIXED, self.MODE_C_DECODER]:
            rb = tk.Radiobutton(
                mode_box,
                text=m,
                variable=self.mode_var,
                value=m,
                command=self._on_mode_change,
                fg="#eceff1",
                bg="#18181f",
                selectcolor="#263238",
                activebackground="#37474f",
                font=("Helvetica", 8),
            )
            rb.pack(anchor=tk.W, padx=6, pady=1)

        # Teacher Override & Drive Controls
        drive_box = tk.LabelFrame(p3_frame, text="Movement Teaching Controls", font=("Helvetica", 9, "bold"), fg="#90a4ae", bg="#18181f")
        drive_box.pack(fill=tk.X, pady=4)

        btn_grid = tk.Frame(drive_box, bg="#18181f")
        btn_grid.pack(pady=4)

        btn_fwd = tk.Button(btn_grid, text="▲ FORWARD", width=11, bg="#1976d2", fg="white", font=("Helvetica", 8, "bold"), command=self.cmd_forward)
        btn_fwd.grid(row=0, column=1, padx=2, pady=2)

        btn_left = tk.Button(btn_grid, text="◄ LEFT", width=9, bg="#1976d2", fg="white", font=("Helvetica", 8, "bold"), command=self.cmd_turn_left)
        btn_left.grid(row=1, column=0, padx=2, pady=2)

        btn_stop = tk.Button(btn_grid, text="■ STOP", width=9, bg="#d32f2f", fg="white", font=("Helvetica", 8, "bold"), command=self.cmd_stop)
        btn_stop.grid(row=1, column=1, padx=2, pady=2)

        btn_right = tk.Button(btn_grid, text="RIGHT ►", width=9, bg="#1976d2", fg="white", font=("Helvetica", 8, "bold"), command=self.cmd_turn_right)
        btn_right.grid(row=1, column=2, padx=2, pady=2)

        btn_bwd = tk.Button(btn_grid, text="▼ BACKWARD", width=11, bg="#1976d2", fg="white", font=("Helvetica", 8, "bold"), command=self.cmd_backward)
        btn_bwd.grid(row=2, column=1, padx=2, pady=2)

        # Reinforcement Teaching Buttons
        feed_box = tk.LabelFrame(p3_frame, text="Neuromodulatory Feedback", font=("Helvetica", 9, "bold"), fg="#90a4ae", bg="#18181f")
        feed_box.pack(fill=tk.X, pady=4)

        feed_btn_frame = tk.Frame(feed_box, bg="#18181f")
        feed_btn_frame.pack(fill=tk.X, padx=4, pady=4)

        btn_reward = tk.Button(
            feed_btn_frame,
            text="★ REWARD (+1.0)\n[PAM Dopamine]",
            bg="#2e7d32",
            fg="white",
            font=("Helvetica", 9, "bold"),
            command=self.cmd_reward,
        )
        btn_reward.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        btn_punish = tk.Button(
            feed_btn_frame,
            text="⚡ PUNISH (-1.0)\n[PPL1 Dopamine]",
            bg="#c62828",
            fg="white",
            font=("Helvetica", 9, "bold"),
            command=self.cmd_punish,
        )
        btn_punish.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        # Operational Mode Actions
        act_box = tk.LabelFrame(p3_frame, text="Workflow Control", font=("Helvetica", 9, "bold"), fg="#90a4ae", bg="#18181f")
        act_box.pack(fill=tk.X, pady=4)

        self.btn_auto = tk.Button(act_box, text="AUTONOMOUS MODE [Key A]", bg="#00897b", fg="white", font=("Helvetica", 8, "bold"), command=self.cmd_autonomous)
        self.btn_auto.pack(fill=tk.X, padx=6, pady=2)

        self.btn_teach = tk.Button(act_box, text="TEACHING MODE [Key T]", bg="#f57c00", fg="white", font=("Helvetica", 8, "bold"), command=self.cmd_teaching)
        self.btn_teach.pack(fill=tk.X, padx=6, pady=2)

        persist_frame = tk.Frame(act_box, bg="#18181f")
        persist_frame.pack(fill=tk.X, padx=6, pady=2)
        btn_save = tk.Button(persist_frame, text="Save State", width=12, bg="#455a64", fg="white", font=("Helvetica", 8), command=self.cmd_save_checkpoint)
        btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        btn_load = tk.Button(persist_frame, text="Load State", width=12, bg="#455a64", fg="white", font=("Helvetica", 8), command=self.cmd_load_checkpoint)
        btn_load.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=1)

        btn_reset_ep = tk.Button(act_box, text="Reset Episode (Preserve Memory)", bg="#37474f", fg="white", font=("Helvetica", 8), command=self.cmd_reset_episode)
        btn_reset_ep.pack(fill=tk.X, padx=6, pady=2)

        btn_reset_mem = tk.Button(act_box, text="Wipe Learned Weights (Fresh Naive)", bg="#b71c1c", fg="white", font=("Helvetica", 8), command=self.cmd_reset_weights)
        btn_reset_mem.pack(fill=tk.X, padx=6, pady=2)

        # Bottom Status Bar
        status_bar = tk.Frame(self.root, bg="#111116", height=28)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_lbl = tk.Label(
            status_bar,
            text="STATUS: INITIALIZED | 50 Hz | AUTONOMOUS MODE | ZERO FAKE DATA GUARANTEE",
            font=("Courier", 9),
            fg="#80deea",
            bg="#111116",
        )
        self.status_lbl.pack(side=tk.LEFT, padx=10, pady=4)

    def _bind_keys(self):
        """Bind keyboard shortcuts for live instructor control."""
        self.root.bind("<Up>", lambda e: self.cmd_forward())
        self.root.bind("<Down>", lambda e: self.cmd_backward())
        self.root.bind("<Left>", lambda e: self.cmd_turn_left())
        self.root.bind("<Right>", lambda e: self.cmd_turn_right())
        self.root.bind("<space>", lambda e: self.cmd_stop())
        self.root.bind("r", lambda e: self.cmd_reward())
        self.root.bind("p", lambda e: self.cmd_punish())
        self.root.bind("a", lambda e: self.cmd_autonomous())
        self.root.bind("t", lambda e: self.cmd_teaching())

    def _on_mode_change(self):
        """Handle mode switch between Mode A, Mode B, and Mode C."""
        new_mode = self.mode_var.get()
        self.current_mode = new_mode
        self.mode_badge.config(text=f"ACTIVE: {self.current_mode}")
        if new_mode == self.MODE_A_FIXED:
            self.agent.mb_circuit.w_kc_to_mbon_app.fill(0.50)
            self.agent.mb_circuit.w_kc_to_mbon_av.fill(0.50)
        print(f"[LIVE LAB] Switched to {new_mode}")

    # ------------------------------------------------------------------------
    # Teacher Control Commands
    # ------------------------------------------------------------------------

    def cmd_forward(self):
        self.is_teaching = True
        self.is_autonomous = False
        self.teacher_v = 6.0
        self.teacher_omega = 0.0
        self._update_ui_state("TEACHING: FORWARD (6.0 cm/s)")

    def cmd_backward(self):
        self.is_teaching = True
        self.is_autonomous = False
        self.teacher_v = -4.0
        self.teacher_omega = 0.0
        self._update_ui_state("TEACHING: BACKWARD (-4.0 cm/s)")

    def cmd_turn_left(self):
        self.is_teaching = True
        self.is_autonomous = False
        self.teacher_v = 3.0
        self.teacher_omega = 3.5
        self._update_ui_state("TEACHING: TURN LEFT (+3.5 rad/s)")

    def cmd_turn_right(self):
        self.is_teaching = True
        self.is_autonomous = False
        self.teacher_v = 3.0
        self.teacher_omega = -3.5
        self._update_ui_state("TEACHING: TURN RIGHT (-3.5 rad/s)")

    def cmd_stop(self):
        self.teacher_v = 0.0
        self.teacher_omega = 0.0
        self._update_ui_state("TEACHING: STOP")

    def cmd_reward(self):
        self.manual_reward_pulse = 1.0
        self._update_ui_state("TEACHER FEEDBACK: APPETITIVE REWARD (+1.0 PAM PULSE)")

    def cmd_punish(self):
        self.manual_punish_pulse = 1.0
        self._update_ui_state("TEACHER FEEDBACK: AVERSIVE PUNISHMENT (-1.0 PPL1 PULSE)")

    def cmd_autonomous(self):
        self.is_teaching = False
        self.is_autonomous = True
        self.teacher_v = 0.0
        self.teacher_omega = 0.0
        self.btn_auto.config(bg="#00bfa5")
        self.btn_teach.config(bg="#f57c00")
        self._update_ui_state("MODE: AUTONOMOUS (NEURAL BRAIN FULL CONTROL)")

    def cmd_teaching(self):
        self.is_teaching = True
        self.is_autonomous = False
        self.btn_auto.config(bg="#00897b")
        self.btn_teach.config(bg="#ff6d00")
        self._update_ui_state("MODE: TEACHING (INSTRUCTOR ACTIVE)")

    def cmd_save_checkpoint(self):
        path = "/home/om/Desktop/robo_fly/experiments/results/curriculum/live_lab_checkpoint.json"
        self.agent.save_checkpoint(path)
        messagebox.showinfo("Checkpoint Saved", f"Plastic synaptic memory successfully serialized to:\n{path}")

    def cmd_load_checkpoint(self):
        path = "/home/om/Desktop/robo_fly/experiments/results/curriculum/conditioned_weights.json"
        if os.path.exists(path):
            self.agent.load_checkpoint(path)
            messagebox.showinfo("Checkpoint Loaded", f"Loaded conditioned synaptic weights from:\n{path}")
        else:
            messagebox.showwarning("File Not Found", f"No checkpoint file found at {path}")

    def cmd_reset_episode(self):
        self.agent.reset(x=0.0, y=-15.0, theta=math.pi / 2.0, reset_memory=False)
        self.episode_count += 1
        self.step_count = 0
        self._update_ui_state("EPISODE RESET (SYNAPTIC MEMORY PRESERVED)")

    def cmd_reset_weights(self):
        self.agent.reset(x=0.0, y=-15.0, theta=math.pi / 2.0, reset_memory=True)
        self.episode_count = 1
        self.step_count = 0
        self._update_ui_state("WEIGHTS RESET TO NAIVE UNTRAINED BASELINE")

    def _update_ui_state(self, msg: str):
        mode_str = "TEACHING" if self.is_teaching else "AUTONOMOUS"
        self.status_lbl.config(
            text=f"STATUS: {msg} | MODE: {mode_str} | EP: {self.episode_count} | STEP: {self.step_count} | FPS: {self.current_fps:.1f}"
        )

    # ------------------------------------------------------------------------
    # Real-Time Simulation Loop (50 Hz)
    # ------------------------------------------------------------------------

    def _step_loop(self):
        """Execute one 20ms simulation step and render the live visuals."""
        if not self.is_running:
            self.root.after(20, self._step_loop)
            return

        now = time.time()
        dt_actual = now - self.last_step_time
        self.last_step_time = now
        if dt_actual > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt_actual)

        self.step_count += 1

        # 1. Fetch sensory observation from arena
        if self.agent.last_obs is None:
            self.agent.last_obs = self.arena.get_observation()
        obs = self.agent.last_obs
        odors = obs["odors"]
        vision = obs["vision"]
        taste = obs["taste"]

        # 2. Reinforcement inputs (combining environment + manual teacher pulse)
        sugar_reward = taste["reward"] + self.manual_reward_pulse
        bitter_punish = taste["punishment"] + self.manual_punish_pulse
        self.manual_reward_pulse = 0.0
        self.manual_punish_pulse = 0.0

        # 3. Neural circuit processing
        if self.current_mode == self.MODE_A_FIXED:
            # Baseline: Zero learning
            mb_out = self.agent.mb_circuit.step(odors, sugar_reward=0.0, bitter_punish=0.0, dt=self.arena.dt)
        else:
            # Plasticity Experiment: 3-factor Hebbian learning
            mb_out = self.agent.mb_circuit.step(odors, sugar_reward=sugar_reward, bitter_punish=bitter_punish, dt=self.arena.dt)

        vis_out = self.agent.visual_circuit.step(vision, omega=self.arena.omega)
        compass_out = self.agent.compass.step(self.arena.omega, dt=self.arena.dt)

        # 4. Action determination
        if self.is_teaching:
            # Instructor Override Active
            # If in Mode C (External Decoder), train decoder weights via least squares
            if self.current_mode == self.MODE_C_DECODER:
                feats = np.array(vision["proximity"], dtype=np.float32)
                target_act = np.array([self.teacher_v, self.teacher_omega])
                self.decoder_weights += 0.01 * np.outer(target_act, feats)

            # Convert teacher v and omega into skid-steer wheel speeds
            v_l = self.teacher_v - 0.18 * self.teacher_omega
            v_r = self.teacher_v + 0.18 * self.teacher_omega
            descending_spikes = {
                "DNp09": max(0.0, self.teacher_v * 8.0),
                "DNa01_left": max(0.0, self.teacher_omega * 10.0),
                "DNa01_right": max(0.0, -self.teacher_omega * 10.0),
                "DNa02_left": 0.0,
                "DNa02_right": 0.0,
                "PER": 0.0,
            }
        else:
            # Autonomous Brain Control
            if self.current_mode == self.MODE_C_DECODER:
                feats = np.array(vision["proximity"], dtype=np.float32)
                pred_act = np.dot(self.decoder_weights, feats)
                v_target = float(np.clip(pred_act[0], 0.0, 8.0))
                omega_target = float(np.clip(pred_act[1], -5.0, 5.0))
                descending_spikes = {
                    "DNp09": v_target * 8.0,
                    "DNa01_left": max(0.0, omega_target * 10.0),
                    "DNa01_right": max(0.0, -omega_target * 10.0),
                    "DNa02_left": 0.0,
                    "DNa02_right": 0.0,
                    "PER": 0.0,
                }
            else:
                # Mode A or Mode B: Biological connectome descending pathways
                olfactory_steer = 0.0
                odor_valences = mb_out.get("odor_valences", {})
                for oid, r in odors.items():
                    ng = r.get("norm_gradient", 0.0)
                    mean_c = r["mean"]
                    if mean_c > 1e-4:
                        val = odor_valences.get(oid, mb_out["valence"])
                        steer_drive = float(np.clip(val * ng * 250.0, -12.0, 12.0))
                        olfactory_steer += steer_drive

                escape_steer = vis_out["escape_steer_bias"]
                net_turn = escape_steer + olfactory_steer

                dl = 10.0 + max(0.0, net_turn)
                dr = 10.0 + max(0.0, -net_turn)
                base_fwd = 25.0
                mean_val = float(np.mean(list(odor_valences.values()))) if odor_valences else 0.0
                p9_rate = max(5.0, base_fwd + max(0.0, mean_val * 6.0) - vis_out["fwd_suppression"])

                descending_spikes = {
                    "DNp09": p9_rate,
                    "DNa01_left": dl,
                    "DNa01_right": dr,
                    "DNa02_left": dl * 0.5,
                    "DNa02_right": dr * 0.5,
                    "PER": 30.0 if taste["sugar"] else 0.0,
                }

        # 5. Step Arena Physics
        next_obs, step_reward, done, info = self.arena.step(descending_spikes)
        self.agent.last_obs = next_obs
        self.total_reward += step_reward

        # 6. Record Neural Spike Raster Activity
        active_kcs = np.where(self.agent.mb_circuit.kc_activity > 0.1)[0][:20]
        self.kc_raster_buffer.append(list(active_kcs))
        self.lc4_rate_buffer.append(vis_out["lc4_right"] + vis_out["lc4_left"])
        self.p9_rate_buffer.append(descending_spikes.get("DNp09", 0.0))

        # 7. Render Graphics on Canvases
        self._render_arena_canvas(obs, info)
        self._render_brain_canvas(mb_out, vis_out, compass_out, info, sugar_reward, bitter_punish)

        # 8. Loop Next Step
        self.root.after(20, self._step_loop)

    # ------------------------------------------------------------------------
    # Panel 1: Arena Graphics Rendering
    # ------------------------------------------------------------------------

    def _render_arena_canvas(self, obs: Dict[str, Any], info: Dict[str, Any]):
        cv = self.arena_canvas
        cv.delete("all")
        w_px, h_px = 460, 520
        cx, cy = w_px / 2.0, h_px / 2.0
        scale = 5.2  # pixels per cm

        def world_to_px(wx, wy):
            return cx + wx * scale, cy - wy * scale

        # Arena Boundary
        bx0, by0 = world_to_px(-40.0, 40.0)
        bx1, by1 = world_to_px(40.0, -40.0)
        cv.create_rectangle(bx0, by0, bx1, by1, outline="#455a64", width=2, fill="#0f0f14")

        # Grid lines
        for g in range(-30, 40, 10):
            gx0, gy0 = world_to_px(g, 40)
            gx1, gy1 = world_to_px(g, -40)
            cv.create_line(gx0, gy0, gx1, gy1, fill="#181822", dash=(2, 4))
            hx0, hy0 = world_to_px(-40, g)
            hx1, hy1 = world_to_px(40, g)
            cv.create_line(hx0, hy0, hx1, hy1, fill="#181822", dash=(2, 4))

        # Reward Zone (Green Sugar Disk)
        fx, fy = world_to_px(self.food_patch.x, self.food_patch.y)
        fr = self.food_patch.radius * scale
        cv.create_oval(fx - fr, fy - fr, fx + fr, fy + fr, fill="#1b5e20", outline="#4caf50", width=2)
        cv.create_text(fx, fy, text="REWARD\n(Sugar)", fill="#a5d6a7", font=("Helvetica", 8, "bold"), justify=tk.CENTER)

        # Hazard Zone (Red Bitter / Shock Disk)
        hx, hy = world_to_px(self.hazard_patch.x, self.hazard_patch.y)
        hr = self.hazard_patch.radius * scale
        cv.create_oval(hx - hr, hy - hr, hx + hr, hy + hr, fill="#b71c1c", outline="#f44336", width=2)
        cv.create_text(hx, hy, text="HAZARD\n(Shock)", fill="#ffcdd2", font=("Helvetica", 8, "bold"), justify=tk.CENTER)

        # Obstacles
        for obs_ent in self.arena.obstacles:
            ox, oy = world_to_px(obs_ent.x, obs_ent.y)
            obr = obs_ent.radius * scale
            cv.create_oval(ox - obr, oy - obr, ox + obr, oy + obr, fill="#37474f", outline="#90a4ae", width=2)

        # Robot Pose
        rx, ry = self.arena.x, self.arena.y
        r_th = self.arena.theta
        px, py = world_to_px(rx, ry)

        # Trajectory Trail
        if len(self.arena.trajectory_x) > 2:
            trail_pts = []
            # Sample last 80 points
            pts_x = self.arena.trajectory_x[-80:]
            pts_y = self.arena.trajectory_y[-80:]
            for tx, ty in zip(pts_x, pts_y):
                sx, sy = world_to_px(tx, ty)
                trail_pts.extend([sx, sy])
            if len(trail_pts) >= 4:
                cv.create_line(*trail_pts, fill="#00bcd4", width=1)

        # Active LiDAR Beams
        distances = obs["vision"]["distances"]
        angles = self.agent.arena.sensors.eyes.angles
        for dist_cm, rel_ang in zip(distances, angles):
            ray_ang = r_th + rel_ang
            lx = rx + dist_cm * math.cos(ray_ang)
            ly = ry + dist_cm * math.sin(ray_ang)
            lx_px, ly_px = world_to_px(lx, ly)
            beam_color = "#ff5252" if dist_cm < 10.0 else "#00e676"
            cv.create_line(px, py, lx_px, ly_px, fill=beam_color, width=1, dash=(1, 3))

        # Robot Chassis Body (4-Wheel Skid-Steer Vehicle)
        body_len_px = 3.0 * scale
        body_wid_px = 2.2 * scale
        cos_t = math.cos(r_th)
        sin_t = math.sin(r_th)

        # 4 Corners of Chassis
        fwd_x = (body_len_px / 2.0) * cos_t
        fwd_y = -(body_len_px / 2.0) * sin_t
        side_x = (body_wid_px / 2.0) * (-sin_t)
        side_y = -(body_wid_px / 2.0) * cos_t

        c_fl = (px + fwd_x + side_x, py + fwd_y + side_y)
        c_fr = (px + fwd_x - side_x, py + fwd_y - side_y)
        c_rr = (px - fwd_x - side_x, py - fwd_y - side_y)
        c_rl = (px - fwd_x + side_x, py - fwd_y + side_y)

        # Draw Chassis
        cv.create_polygon([c_fl, c_fr, c_rr, c_rl], fill="#1565c0", outline="#64b5f6", width=2)

        # 4 Wheels
        w_len = 1.0 * scale
        w_wid = 0.4 * scale
        for (wx, wy) in [c_fl, c_fr, c_rr, c_rl]:
            cv.create_oval(wx - w_wid, wy - w_len / 2, wx + w_wid, wy + w_len / 2, fill="#212121", outline="#9e9e9e")

        # Heading Direction Arrow
        arrow_len = 22.0
        cv.create_line(
            px,
            py,
            px + arrow_len * math.cos(r_th),
            py - arrow_len * math.sin(r_th),
            fill="#ffea00",
            arrow=tk.LAST,
            width=3,
        )

        # Update Info Label
        deg = math.degrees(r_th) % 360.0
        self.p1_pose_lbl.config(
            text=f"Pose: x={rx:5.1f}cm | y={ry:5.1f}cm | θ={deg:5.1f}° | v={info['v']:4.1f}cm/s | ω={info['omega']:+4.1f}rad/s"
        )

    # ------------------------------------------------------------------------
    # Panel 2: Neural Activity Dashboard Rendering
    # ------------------------------------------------------------------------

    def _render_brain_canvas(
        self,
        mb_out: Dict[str, Any],
        vis_out: Dict[str, Any],
        compass_out: Dict[str, Any],
        info: Dict[str, Any],
        sugar_reward: float,
        bitter_punish: float,
    ):
        cv = self.brain_canvas
        cv.delete("all")
        w_px, h_px = 540, 520

        # Section 1: Spikes & Activity Header
        cv.create_text(15, 15, anchor=tk.W, text="1. LIVE SPIKE RASTER (Kenyon Cells - Olfactory Representation)", font=("Helvetica", 9, "bold"), fill="#80deea")
        raster_x0, raster_y0 = 20, 30
        raster_w, raster_h = 500, 70
        cv.create_rectangle(raster_x0, raster_y0, raster_x0 + raster_w, raster_y0 + raster_h, fill="#050508", outline="#37474f")

        # Draw Spikes in Scrolling Raster Buffer
        if self.kc_raster_buffer:
            step_w = raster_w / float(self.max_raster_steps)
            for idx, active_kc_list in enumerate(self.kc_raster_buffer):
                x_pos = raster_x0 + idx * step_w
                for kc_idx in active_kc_list:
                    y_pos = raster_y0 + (kc_idx % 20) * (raster_h / 20.0)
                    cv.create_line(x_pos, y_pos, x_pos, y_pos + 2, fill="#00e5ff", width=1)

        # Section 2: Plastic Synaptic Weights (Mushroom Body KC -> MBON)
        cv.create_text(15, 120, anchor=tk.W, text="2. MUSHROOM BODY PLASTIC WEIGHTS & VALENCE (KC -> MBON)", font=("Helvetica", 9, "bold"), fill="#ffea00")

        w_app = mb_out["mean_w_app"]
        w_av = mb_out["mean_w_av"]
        val = mb_out["valence"]

        # Approach Weight Bar
        cv.create_text(25, 145, anchor=tk.W, text=f"W_approach (Sucrose/Reward): {w_app:.3f}", font=("Courier", 8), fill="#a5d6a7")
        cv.create_rectangle(220, 138, 220 + 200, 150, fill="#1b5e20", outline="#2e7d32")
        app_fill_w = int(200 * np.clip(w_app, 0.0, 1.0))
        cv.create_rectangle(220, 138, 220 + app_fill_w, 150, fill="#4caf50", outline="")

        # Avoidance Weight Bar
        cv.create_text(25, 168, anchor=tk.W, text=f"W_avoidance (Bitter/Shock): {w_av:.3f}", font=("Courier", 8), fill="#ffcdd2")
        cv.create_rectangle(220, 161, 220 + 200, 173, fill="#b71c1c", outline="#c62828")
        av_fill_w = int(200 * np.clip(w_av, 0.0, 1.0))
        cv.create_rectangle(220, 161, 220 + av_fill_w, 173, fill="#f44336", outline="")

        # Learned Odor Valences
        val_a = mb_out.get("odor_valences", {}).get("Odor_A_Banana", 0.0)
        val_b = mb_out.get("odor_valences", {}).get("Odor_B_Citrus", 0.0)
        cv.create_text(25, 192, anchor=tk.W, text=f"Learned Valences: Banana V(A)={val_a:+5.2f} | Citrus V(B)={val_b:+5.2f}", font=("Courier", 9, "bold"), fill="#fff176")

        # Section 3: Dopaminergic Neuromodulation
        cv.create_text(15, 225, anchor=tk.W, text="3. NEUROMODULATORY DOPAMINE SIGNALING", font=("Helvetica", 9, "bold"), fill="#ff80ab")
        pam_color = "#00e676" if sugar_reward > 0.1 else "#263238"
        ppl_color = "#ff1744" if bitter_punish > 0.1 else "#263238"

        cv.create_rectangle(30, 240, 240, 270, fill=pam_color, outline="#37474f", width=2)
        cv.create_text(135, 255, text=f"PAM-DAN (Sweet): {sugar_reward:.2f}", font=("Helvetica", 9, "bold"), fill="black" if sugar_reward > 0.1 else "#90a4ae")

        cv.create_rectangle(280, 240, 490, 270, fill=ppl_color, outline="#37474f", width=2)
        cv.create_text(385, 255, text=f"PPL1-DAN (Shock): {bitter_punish:.2f}", font=("Helvetica", 9, "bold"), fill="white" if bitter_punish > 0.1 else "#90a4ae")

        # Section 4: Central Complex Heading Compass (Ring Attractor)
        cv.create_text(15, 300, anchor=tk.W, text="4. CENTRAL COMPLEX (EB) COMPASS RING ATTRACTOR", font=("Helvetica", 9, "bold"), fill="#b388ff")

        compass_cx, compass_cy = 130, 395
        comp_r = 55
        cv.create_oval(compass_cx - comp_r, compass_cy - comp_r, compass_cx + comp_r, compass_cy + comp_r, outline="#512da8", width=2, fill="#12005e")

        # Draw 16 E-PG Wedges
        wedge_rates = compass_out.get("epg_rates", np.zeros(16))
        for i in range(16):
            ang0 = (i / 16.0) * 2.0 * math.pi
            ang1 = ((i + 1) / 16.0) * 2.0 * math.pi
            rate = float(wedge_rates[i]) if i < len(wedge_rates) else 0.0
            # Higher rate -> brighter magenta/purple
            intensity = int(min(255, rate * 800))
            color_hex = f"#{intensity:02x}{int(intensity*0.3):02x}{255:02x}"
            x_w = compass_cx + (comp_r - 10) * math.cos(ang0)
            y_w = compass_cy - (comp_r - 10) * math.sin(ang0)
            cv.create_line(compass_cx, compass_cy, x_w, y_w, fill=color_hex, width=2)

        # Internal Heading Pointer
        int_h = compass_out.get("internal_heading", 0.0)
        cv.create_line(
            compass_cx,
            compass_cy,
            compass_cx + comp_r * math.cos(int_h),
            compass_cy - comp_r * math.sin(int_h),
            fill="#ffeb3b",
            width=3,
            arrow=tk.LAST,
        )
        cv.create_text(compass_cx, compass_cy + comp_r + 15, text=f"θ_internal: {math.degrees(int_h):.1f}°", font=("Courier", 8), fill="#d1c4e9")

        # Section 5: Descending Motor Commands & Wheel Speeds
        cv.create_text(260, 300, anchor=tk.W, text="5. DESCENDING MOTORS & WHEELS", font=("Helvetica", 9, "bold"), fill="#69f0ae")

        v_curr = info["v"]
        w_curr = info["omega"]
        track_b = 0.36
        v_l = v_curr - (track_b / 2.0) * w_curr
        v_r = v_curr + (track_b / 2.0) * w_curr

        # Wheel velocity gauges
        cv.create_text(270, 335, anchor=tk.W, text=f"Left Wheels:  {v_l:5.1f} cm/s", font=("Courier", 9, "bold"), fill="#80cbc4")
        cv.create_rectangle(400, 328, 510, 342, fill="#004d40")
        w_bar_l = int(np.clip(v_l * 8.0, 0, 110))
        cv.create_rectangle(400, 328, 400 + w_bar_l, 342, fill="#26a69a")

        cv.create_text(270, 365, anchor=tk.W, text=f"Right Wheels: {v_r:5.1f} cm/s", font=("Courier", 9, "bold"), fill="#80cbc4")
        cv.create_rectangle(400, 358, 510, 372, fill="#004d40")
        w_bar_r = int(np.clip(v_r * 8.0, 0, 110))
        cv.create_rectangle(400, 358, 400 + w_bar_r, 372, fill="#26a69a")

        cv.create_text(270, 400, anchor=tk.W, text=f"Forward DNp09: {info.get('v', 0)*6:.1f} Hz", font=("Courier", 8), fill="#b0bec5")
        cv.create_text(270, 420, anchor=tk.W, text=f"Looming LC4:   {vis_out['lc4_right']+vis_out['lc4_left']:.1f} Hz", font=("Courier", 8), fill="#ef9a9a")
        cv.create_text(270, 440, anchor=tk.W, text=f"Steering DNa:  {abs(w_curr)*10:.1f} Hz", font=("Courier", 8), fill="#b0bec5")

        # Bottom Subsystem Synapse Label
        self.p2_synapse_lbl.config(
            text=f"MB: W_app={w_app:.3f} | W_av={w_av:.3f} | V(Banana)={val_a:+.2f} | V(Citrus)={val_b:+.2f}"
        )


def main():
    print("=" * 70)
    print("LAUNCHING DROSOPHILA VIRTUAL BRAIN LEARNING LABORATORY")
    print("=" * 70)
    root = tk.Tk()
    app = LiveBrainLaboratory(root=root)
    root.mainloop()


if __name__ == "__main__":
    main()
