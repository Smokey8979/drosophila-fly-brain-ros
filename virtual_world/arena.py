"""Continuous 2D embodied arena for Drosophila connectome simulation."""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .sensors import SensorySystem
from .actuators import MotorActuator


class OdorSource:
    """Represents a chemical odor emitter in the arena."""

    def __init__(
        self,
        odor_id: str,
        x: float,
        y: float,
        peak_concentration: float = 1.0,
        sigma: float = 15.0,
        active: bool = True,
    ):
        self.odor_id = odor_id
        self.x = x
        self.y = y
        self.peak_concentration = peak_concentration
        self.sigma = sigma
        self.active = active

    def get_concentration(self, px: float, py: float) -> float:
        if not self.active:
            return 0.0
        d_sq = (px - self.x) ** 2 + (py - self.y) ** 2
        return float(self.peak_concentration * math.exp(-d_sq / (2.0 * self.sigma * self.sigma)))


class FoodPatch:
    """Represents a nutritious substrate (e.g., sucrose drop) associated with an odor."""

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 3.0,
        reward_value: float = 1.0,
        odor_id: Optional[str] = None,
    ):
        self.x = x
        self.y = y
        self.radius = radius
        self.reward_value = reward_value
        self.odor_id = odor_id
        self.consumed_count = 0


class HazardPatch:
    """Represents an aversive zone (e.g., bitter quinine / electric shock)."""

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 3.0,
        penalty_value: float = 1.0,
        odor_id: Optional[str] = None,
    ):
        self.x = x
        self.y = y
        self.radius = radius
        self.penalty_value = penalty_value
        self.odor_id = odor_id
        self.shock_count = 0


class Obstacle:
    """Circular physical barrier or landmark."""

    def __init__(self, x: float, y: float, radius: float = 4.0):
        self.x = x
        self.y = y
        self.radius = radius


class FlyArena:
    """High-performance continuous simulation arena for the embodied fly."""

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        dt: float = 0.02,
        seed: Optional[int] = None,
    ):
        """
        Args:
            width: Total arena width in cm (default 100 cm, [-50, 50]).
            height: Total arena height in cm (default 100 cm, [-50, 50]).
            dt: Environment timestep in seconds (default 20 ms).
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        self.width = width
        self.height = height
        self.dt = dt
        self.x_bounds = (-width / 2.0, width / 2.0)
        self.y_bounds = (-height / 2.0, height / 2.0)

        # Entity collections
        self.odor_sources: List[OdorSource] = []
        self.food_patches: List[FoodPatch] = []
        self.hazard_patches: List[HazardPatch] = []
        self.obstacles: List[Obstacle] = []

        # Sensory and motor submodules
        self.sensors = SensorySystem()
        self.actuator = MotorActuator()

        # Agent state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0

        # Metrics and tracking
        self.trajectory_x: List[float] = []
        self.trajectory_y: List[float] = []
        self.trajectory_theta: List[float] = []
        self.step_count = 0
        self.total_reward = 0.0
        self.total_penalty = 0.0
        self.food_encounters = 0
        self.shock_encounters = 0

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Reset agent pose and episode tracking."""
        self.x = float(x)
        self.y = float(y)
        self.theta = float(np.random.uniform(-math.pi, math.pi) if theta is None else theta)
        self.v = 0.0
        self.omega = 0.0
        self.actuator.reset()
        self.sensors.eyes.reset()

        self.trajectory_x = [self.x]
        self.trajectory_y = [self.y]
        self.trajectory_theta = [self.theta]
        self.step_count = 0
        self.total_reward = 0.0
        self.total_penalty = 0.0
        self.food_encounters = 0
        self.shock_encounters = 0

        for food in self.food_patches:
            food.consumed_count = 0
        for haz in self.hazard_patches:
            haz.shock_count = 0

        return self.get_observation()

    def add_odor(self, odor_id: str, x: float, y: float, peak: float = 1.0, sigma: float = 15.0) -> OdorSource:
        src = OdorSource(odor_id, x, y, peak, sigma)
        self.odor_sources.append(src)
        return src

    def add_food(self, x: float, y: float, radius: float = 3.0, reward: float = 1.0, odor_id: Optional[str] = None) -> FoodPatch:
        patch = FoodPatch(x, y, radius, reward, odor_id)
        self.food_patches.append(patch)
        return patch

    def add_hazard(self, x: float, y: float, radius: float = 3.0, penalty: float = 1.0, odor_id: Optional[str] = None) -> HazardPatch:
        patch = HazardPatch(x, y, radius, penalty, odor_id)
        self.hazard_patches.append(patch)
        return patch

    def add_obstacle(self, x: float, y: float, radius: float = 4.0) -> Obstacle:
        obs = Obstacle(x, y, radius)
        self.obstacles.append(obs)
        return obs

    def clear_entities(self):
        """Remove all arena entities."""
        self.odor_sources.clear()
        self.food_patches.clear()
        self.hazard_patches.clear()
        self.obstacles.clear()

    def get_observation(self) -> Dict[str, Any]:
        """Query sensory system for current state."""
        arena_bounds = (self.x_bounds[0], self.x_bounds[1], self.y_bounds[0], self.y_bounds[1])
        return self.sensors.observe(
            self.x,
            self.y,
            self.theta,
            arena_bounds,
            self.odor_sources,
            self.obstacles,
            self.food_patches,
            self.hazard_patches,
            dt=self.dt,
        )

    def step(self, descending_spikes: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance simulation by one timestep using descending neural commands.
        
        Args:
            descending_spikes: Dict of descending spike rates.
            
        Returns:
            Tuple of (observation, reward, done, info).
        """
        self.step_count += 1

        # Decode motor commands
        self.v, self.omega, per_extended = self.actuator.decode_action(descending_spikes, dt=self.dt)

        # Update pose kinematics
        self.theta += self.omega * self.dt
        # Normalize theta to [-pi, pi)
        self.theta = (self.theta + math.pi) % (2.0 * math.pi) - math.pi

        new_x = self.x + self.v * math.cos(self.theta) * self.dt
        new_y = self.y + self.v * math.sin(self.theta) * self.dt

        # Boundary collisions
        margin = 1.0
        hit_wall = False
        if new_x < self.x_bounds[0] + margin:
            new_x = self.x_bounds[0] + margin
            self.theta = math.pi - self.theta  # Deflect
            hit_wall = True
        elif new_x > self.x_bounds[1] - margin:
            new_x = self.x_bounds[1] - margin
            self.theta = math.pi - self.theta
            hit_wall = True

        if new_y < self.y_bounds[0] + margin:
            new_y = self.y_bounds[0] + margin
            self.theta = -self.theta
            hit_wall = True
        elif new_y > self.y_bounds[1] - margin:
            new_y = self.y_bounds[1] - margin
            self.theta = -self.theta
            hit_wall = True

        # Obstacle collisions
        for obs in self.obstacles:
            dist = math.hypot(new_x - obs.x, new_y - obs.y)
            min_dist = obs.radius + 0.5
            if dist < min_dist and dist > 1e-4:
                push_dir_x = (new_x - obs.x) / dist
                push_dir_y = (new_y - obs.y) / dist
                new_x = obs.x + push_dir_x * min_dist
                new_y = obs.y + push_dir_y * min_dist
                self.theta = math.atan2(push_dir_y, push_dir_x) + np.random.uniform(-0.5, 0.5)

        self.x = new_x
        self.y = new_y
        self.trajectory_x.append(self.x)
        self.trajectory_y.append(self.y)
        self.trajectory_theta.append(self.theta)

        # Gather observation
        obs = self.get_observation()
        taste = obs["taste"]
        step_reward = taste["net_reinforcement"]
        self.total_reward += taste["reward"]
        self.total_penalty += taste["punishment"]

        if taste["sugar"]:
            self.food_encounters += 1
            if taste["active_food"]:
                taste["active_food"].consumed_count += 1

        if taste["bitter"]:
            self.shock_encounters += 1
            if taste["active_hazard"]:
                taste["active_hazard"].shock_count += 1

        info = {
            "step": self.step_count,
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "v": self.v,
            "omega": self.omega,
            "hit_wall": hit_wall,
            "food_encounters": self.food_encounters,
            "shock_encounters": self.shock_encounters,
            "total_reward": self.total_reward,
            "total_penalty": self.total_penalty,
        }

        return obs, step_reward, False, info

    def render_trajectory(self, save_path: str, title: str = "Agent Trajectory in FlyArena"):
        """Plot the arena, entities, and agent's movement trail."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.x_bounds[0] - 2, self.x_bounds[1] + 2)
        ax.set_ylim(self.y_bounds[0] - 2, self.y_bounds[1] + 2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Draw boundaries
        rect = plt.Rectangle(
            (self.x_bounds[0], self.y_bounds[0]),
            self.width,
            self.height,
            fill=False,
            color="black",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Draw odor contours
        if self.odor_sources:
            grid_x, grid_y = np.meshgrid(
                np.linspace(self.x_bounds[0], self.x_bounds[1], 100),
                np.linspace(self.y_bounds[0], self.y_bounds[1], 100),
            )
            for src in self.odor_sources:
                if src.active:
                    field = src.peak_concentration * np.exp(
                        -((grid_x - src.x) ** 2 + (grid_y - src.y) ** 2) / (2.0 * src.sigma ** 2)
                    )
                    cmap = "Oranges" if "sugar" in src.odor_id.lower() or "a" in src.odor_id.lower() else "Blues"
                    ax.contour(grid_x, grid_y, field, levels=4, cmap=cmap, alpha=0.35)

        # Draw Food Patches
        for food in self.food_patches:
            circle = plt.Circle((food.x, food.y), food.radius, color="green", alpha=0.4, label="Food (Sugar)")
            ax.add_patch(circle)
            ax.text(food.x, food.y, f"Food ({food.odor_id or ''})", color="darkgreen", ha="center", fontsize=9, fontweight="bold")

        # Draw Hazard Patches
        for haz in self.hazard_patches:
            circle = plt.Circle((haz.x, haz.y), haz.radius, color="red", alpha=0.4, label="Hazard (Bitter)")
            ax.add_patch(circle)
            ax.text(haz.x, haz.y, f"Hazard ({haz.odor_id or ''})", color="darkred", ha="center", fontsize=9, fontweight="bold")

        # Draw Obstacles
        for obs in self.obstacles:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.7)
            ax.add_patch(circle)

        # Draw Agent Trajectory
        if len(self.trajectory_x) > 1:
            ax.plot(self.trajectory_x, self.trajectory_y, color="blue", linewidth=1.5, alpha=0.75, label="Fly Trail")
            # Mark start
            ax.scatter([self.trajectory_x[0]], [self.trajectory_y[0]], color="cyan", s=60, marker="o", label="Start", zorder=5)
            # Mark current/end
            ax.scatter([self.trajectory_x[-1]], [self.trajectory_y[-1]], color="blue", s=80, marker="^", label="Fly Position", zorder=5)

            # Draw heading arrow
            end_x = self.trajectory_x[-1]
            end_y = self.trajectory_y[-1]
            end_theta = self.trajectory_theta[-1]
            arrow_len = 3.0
            ax.arrow(
                end_x,
                end_y,
                arrow_len * math.cos(end_theta),
                arrow_len * math.sin(end_theta),
                head_width=1.5,
                color="navy",
                zorder=6,
            )

        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9, fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
