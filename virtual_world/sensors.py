"""Sensory models for the embodied Drosophila agent."""

import math
from typing import Dict, List, Tuple, Any
import numpy as np


class AntennaeSensor:
    """Bilateral olfactory antennae sensor.
    
    Computes chemical concentration at left and right antennal tips to
    allow spatial gradient detection (tropotaxis).
    """

    def __init__(self, body_length: float = 0.25, antennal_span: float = 0.05):
        """
        Args:
            body_length: Length of fly body in cm.
            antennal_span: Distance between left and right antennal tips in cm.
        """
        self.body_length = body_length
        self.antennal_span = antennal_span

    def get_antennal_positions(self, x: float, y: float, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (x, y) coordinates for left and right antennae."""
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        fwd_x = x + (self.body_length / 2.0) * cos_t
        fwd_y = y + (self.body_length / 2.0) * sin_t
        half_span = self.antennal_span / 2.0

        # Left antenna is at +90 deg relative to heading
        left_pos = np.array([fwd_x - half_span * sin_t, fwd_y + half_span * cos_t])
        # Right antenna is at -90 deg relative to heading
        right_pos = np.array([fwd_x + half_span * sin_t, fwd_y - half_span * cos_t])
        return left_pos, right_pos

    def sample_odors(
        self,
        x: float,
        y: float,
        theta: float,
        odor_sources: List[Any],
        noise_level: float = 0.02
    ) -> Dict[str, Dict[str, float]]:
        """Sample concentrations for all odor sources at left and right antennae.
        
        Returns:
            Dict mapping odor_id -> {'left': conc_left, 'right': conc_right, 'gradient': grad}
        """
        left_pos, right_pos = self.get_antennal_positions(x, y, theta)
        readings = {}

        for src in odor_sources:
            c_left = src.get_concentration(left_pos[0], left_pos[1])
            c_right = src.get_concentration(right_pos[0], right_pos[1])

            if noise_level > 0.0:
                c_left = max(0.0, c_left + np.random.normal(0.0, noise_level * (c_left + 0.01)))
                c_right = max(0.0, c_right + np.random.normal(0.0, noise_level * (c_right + 0.01)))

            gradient = c_left - c_right
            c_mean = 0.5 * (c_left + c_right)
            norm_gradient = gradient / (c_mean + 0.02)
            readings[src.odor_id] = {
                "left": float(c_left),
                "right": float(c_right),
                "mean": float(c_mean),
                "gradient": float(gradient),
                "norm_gradient": float(norm_gradient),
            }
        return readings


class CompoundEyeSensor:
    """Panoramic compound eye visual array.
    
    Simulates multi-ommatidia visual field raycasting to detect obstacle
    proximity, looming motion (LC4 activation), and visual landmarks.
    """

    def __init__(
        self,
        num_ommatidia: int = 16,
        fov_rad: float = 3.0 * math.pi / 2.0,  # 270 degrees FOV
        max_range: float = 30.0,
    ):
        """
        Args:
            num_ommatidia: Number of discrete azimuthal facets.
            fov_rad: Total horizontal field of view in radians.
            max_range: Maximum detection distance in cm.
        """
        self.num_ommatidia = num_ommatidia
        self.fov_rad = fov_rad
        self.max_range = max_range
        # Relative angles for each ommatidium (-fov/2 to +fov/2)
        self.angles = np.linspace(-fov_rad / 2.0, fov_rad / 2.0, num_ommatidia)
        self.prev_distances = np.full(num_ommatidia, max_range, dtype=np.float32)
        self.initialized = False

    def reset(self):
        """Reset raycast distance memory."""
        self.prev_distances.fill(self.max_range)
        self.initialized = False

    def raycast(
        self,
        x: float,
        y: float,
        theta: float,
        arena_bounds: Tuple[float, float, float, float],
        obstacles: List[Any],
        dt: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """Perform 2D raycasting for all ommatidia.
        
        Returns:
            Dict containing:
              - 'distances': Distance to nearest collision for each facet (cm)
              - 'normalized_proximity': 1.0 - (d / max_range) in [0, 1]
              - 'looming': Expansion rate (-dd/dt / d) triggering LC4 avoidance
        """
        x_min, x_max, y_min, y_max = arena_bounds
        distances = np.full(self.num_ommatidia, self.max_range, dtype=np.float32)

        for i, rel_angle in enumerate(self.angles):
            ray_theta = theta + rel_angle
            dx = math.cos(ray_theta)
            dy = math.sin(ray_theta)

            min_dist = self.max_range

            # Intersect with arena boundary walls
            if dx > 1e-6:
                d = (x_max - x) / dx
                if d > 0:
                    min_dist = min(min_dist, d)
            elif dx < -1e-6:
                d = (x_min - x) / dx
                if d > 0:
                    min_dist = min(min_dist, d)

            if dy > 1e-6:
                d = (y_max - y) / dy
                if d > 0:
                    min_dist = min(min_dist, d)
            elif dy < -1e-6:
                d = (y_min - y) / dy
                if d > 0:
                    min_dist = min(min_dist, d)

            # Intersect with circular obstacles
            for obs in obstacles:
                ox, oy = obs.x, obs.y
                r = obs.radius
                # Vector from ray origin to circle center
                fx = x - ox
                fy = y - oy
                b = 2.0 * (fx * dx + fy * dy)
                c = fx * fx + fy * fy - r * r
                disc = b * b - 4.0 * c
                if disc >= 0:
                    sqrt_disc = math.sqrt(disc)
                    t1 = (-b - sqrt_disc) / 2.0
                    if 0.0 < t1 < min_dist:
                        min_dist = t1

            distances[i] = min(min_dist, self.max_range)

        # Compute optical looming expansion rate
        proximity = np.clip(1.0 - (distances / self.max_range), 0.0, 1.0)
        if not self.initialized:
            self.prev_distances = distances.copy()
            self.initialized = True
            dist_change = np.zeros_like(distances)
        else:
            dist_change = (distances - self.prev_distances) / max(dt, 1e-4)

        # Approach rate is physically bounded by locomotion speed (prevents ray-edge jump artifacts)
        approach_rate = np.clip(-dist_change, 0.0, 10.0)
        looming = approach_rate / (distances + 1.0)

        self.prev_distances = distances.copy()

        return {
            "distances": distances,
            "proximity": proximity,
            "looming": looming,
        }


class GustatorySensor:
    """Proboscis contact sensor for taste and reward/punishment feedback.
    
    Evaluates direct physical contact with nutritive substrates (sugar/sucrose)
    or toxic/aversive substrates (quinine/shock).
    """

    def __init__(self, contact_radius: float = 0.5):
        """
        Args:
            contact_radius: Effective tasting distance around proboscis in cm.
        """
        self.contact_radius = contact_radius

    def sample_tastants(
        self,
        proboscis_pos: np.ndarray,
        food_patches: List[Any],
        hazard_patches: List[Any],
    ) -> Dict[str, Any]:
        """Detect gustatory contacts at proboscis tip."""
        px, py = proboscis_pos[0], proboscis_pos[1]
        sugar_contact = False
        bitter_contact = False
        reward_value = 0.0
        punishment_value = 0.0
        active_food = None
        active_hazard = None

        for patch in food_patches:
            dist = math.hypot(px - patch.x, py - patch.y)
            if dist <= (patch.radius + self.contact_radius):
                sugar_contact = True
                reward_value += patch.reward_value
                active_food = patch
                break

        for patch in hazard_patches:
            dist = math.hypot(px - patch.x, py - patch.y)
            if dist <= (patch.radius + self.contact_radius):
                bitter_contact = True
                punishment_value += patch.penalty_value
                active_hazard = patch
                break

        return {
            "sugar": sugar_contact,
            "bitter": bitter_contact,
            "reward": float(reward_value),
            "punishment": float(punishment_value),
            "net_reinforcement": float(reward_value - punishment_value),
            "active_food": active_food,
            "active_hazard": active_hazard,
        }


class SensorySystem:
    """Unified sensory coordinator for the virtual fly agent."""

    def __init__(self, body_length: float = 0.25):
        self.body_length = body_length
        self.antennae = AntennaeSensor(body_length=body_length)
        self.eyes = CompoundEyeSensor(num_ommatidia=16)
        self.gustatory = GustatorySensor()

    def get_proboscis_position(self, x: float, y: float, theta: float) -> np.ndarray:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array([x + (self.body_length / 2.0) * cos_t, y + (self.body_length / 2.0) * sin_t])

    def observe(
        self,
        x: float,
        y: float,
        theta: float,
        arena_bounds: Tuple[float, float, float, float],
        odor_sources: List[Any],
        obstacles: List[Any],
        food_patches: List[Any],
        hazard_patches: List[Any],
        dt: float = 0.01,
    ) -> Dict[str, Any]:
        """Collect all multi-modal observations in a single pass."""
        odors = self.antennae.sample_odors(x, y, theta, odor_sources)
        vision = self.eyes.raycast(x, y, theta, arena_bounds, obstacles, dt=dt)
        proboscis_pos = self.get_proboscis_position(x, y, theta)
        taste = self.gustatory.sample_tastants(proboscis_pos, food_patches, hazard_patches)

        return {
            "odors": odors,
            "vision": vision,
            "taste": taste,
            "agent_pose": {"x": x, "y": y, "theta": theta},
        }
