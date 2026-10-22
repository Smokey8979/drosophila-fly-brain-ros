"""
Standalone Demonstration Utility for the Fly Brain Robot Control Pipeline.

Runs a multi-cycle simulation loop of the 138k-neuron connectome engine
coupled to sensory encoding, motor decoding, safety validation, and simulated
chassis kinematics without requiring an active ROS 2 middleware daemon.
"""

import sys
import os
from pathlib import Path
from time import perf_counter as time, sleep

# Ensure fly_brain_ros is in python path
pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from fly_brain_ros.fly_brain_node import FlyBrainControllerPipeline
from fly_brain_ros.mock_chassis import MockFourWheelChassis


def run_standalone_demo(num_cycles=5, cycle_dt=0.1):
    """Executes a standalone demonstration of the neural control pipeline."""
    print("=" * 70)
    print(" DROSOPHILA CONNECTOME 4-WHEEL ROBOT CONTROL: STANDALONE DEMO")
    print("=" * 70)

    # Resolve default data paths
    workspace_root = pkg_root.parent
    data_dir = workspace_root / 'fly-brain' / 'data'
    comp_path = str(data_dir / '2025_Completeness_783.csv')
    conn_path = str(data_dir / '2025_Connectivity_783.parquet')
    cache_dir = str(data_dir)

    print(f"[DEMO] Loading connectome completeness: {comp_path}")
    print(f"[DEMO] Loading connectome connectivity:   {conn_path}")

    pipeline = FlyBrainControllerPipeline(
        comp_path=comp_path,
        conn_path=conn_path,
        cache_dir=cache_dir,
        dt_ms=0.1,
        steps_per_cycle=10,
        device="cpu"
    )

    chassis = MockFourWheelChassis(x=0.0, y=0.0, yaw=0.0, track_width=0.36, wheel_radius=0.085)

    try:
        pipeline.start()
        print("\n--- Starting Closed-Loop Demonstration Cycles ---")

        for cycle in range(num_cycles):
            t_start = time()

            # 1. Generate LiDAR scan from mock chassis in simulated environment
            scan = chassis.generate_laserscan(num_rays=180, fov_rad=3.14159)
            ranges = scan['ranges']
            angle_min = scan['angle_min']
            angle_inc = scan['angle_increment']

            # Extract sector ranges
            rates_dict = pipeline.encoder.encode_laserscan(ranges, angle_min, angle_inc)
            min_dist = min(ranges)

            # Pass ranges into pipeline
            pipeline.handle_sensor_ranges(min_dist, min_dist, min_dist)

            # Wait for control cycle period
            sleep(cycle_dt)

            # Retrieve command
            cmd, status = pipeline.get_latest_command()

            # Step physical chassis
            chassis.set_cmd_vel(cmd['linear_x'], cmd['angular_z'])
            chassis.step_physics(dt=cycle_dt)

            elapsed = (time() - t_start) * 1000.0

            print(
                f"Cycle {cycle+1:02d}: Status={status:<22} | "
                f"Vx={cmd['linear_x']:+.3f} m/s | Wz={cmd['angular_z']:+.3f} rad/s | "
                f"v_L={cmd['v_left_wheels']:+.3f} m/s | v_R={cmd['v_right_wheels']:+.3f} m/s | "
                f"Pose=({chassis.x:.2f}, {chassis.y:.2f}) | Latency={elapsed:.1f}ms"
            )

        print("\n[DEMO] Completed successfully.")
    finally:
        pipeline.stop()


def main():
    cycles = 5
    if len(sys.argv) > 1:
        try:
            cycles = int(sys.argv[1])
        except ValueError:
            pass
    run_standalone_demo(num_cycles=cycles)


if __name__ == '__main__':
    main()
