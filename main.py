"""
CARLA Data Collection and Simulation Script

This module orchestrates the CARLA simulator to collect sensor data (RGB cameras,
LiDAR) along with object annotations. It manages the simulation loop, sensor data
capture, and asynchronous disk I/O for efficient data storage.
"""

import yaml
import os
import sys
import math
import argparse
from datetime import datetime

sys.path.append("src/")
try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

from src.pygame_display import DisplayManager
from src.world_manager import CarlaWorldManager
from src.utils import capture_data_async, AsyncDiskWriter, compute_K

# Constants
SENSOR_DATA_TIMEOUT = 2.0  # Timeout for retrieving sensor data (seconds)
WEATHER_TICK_FACTOR = 0.20  # Factor to adjust weather update speed
MIN_VELOCITY_THRESHOLD = 0.001  # Minimum velocity to consider vehicle in motion
DISK_WRITER_WORKERS = 8  # Number of async disk writer threads
DISK_WRITER_QUEUE_SIZE = 200  # Maximum queue size for disk writer


def main(config_path):
    """
    Main simulation and data collection loop.
    
    Orchestrates CARLA world setup, sensor configuration, and the primary
    simulation loop. Handles data capture from RGB cameras and LiDAR sensors
    with asynchronous disk writing for efficient I/O.
    
    Args:
        config_path (str): Path to the configuration file (config.yaml).
    
    Raises:
        FileNotFoundError: If config file is not found.
        Exception: Any runtime errors during simulation are caught and logged.
    """
    try:
        # Load configuration file from provided path
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Create output directory with timestamp and scenario info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weather_suffix = "dynamic_weather" if cfg["dynamic_weather"] else cfg["weather"]
        out_dir = os.path.join(cfg["out_dir"], f'run_{cfg["map"]}_{weather_suffix}_{timestamp}')
        os.makedirs(out_dir, exist_ok=True)

        # Initialize CARLA world manager
        carla_world = CarlaWorldManager(cfg=cfg)
        carla_world.spawn_actors()
        carla_world.spawn_peds()
        carla_world.spawn_ego_vehicle()
        
        # Setup visualization display if enabled
        display_man = None
        if cfg["sensor_preview"]:
            grid_size = [
                math.ceil(carla_world.ego_vehicle.num_cameras / 3),
                min(carla_world.ego_vehicle.num_cameras, 3)
            ]
            display_man = DisplayManager(grid_size, window_size=[1280, 720])

        # Configure all sensors
        carla_world.ego_vehicle.sensor_setup(
            carla_world.world, display_man, enable_lidar_vis=cfg["sensor_preview"]
        )
        carla_world.ego_vehicle.ego_vehicle.set_autopilot(True)
        carla_world.set_synchronous()

        # Calculate frame skipping for data capture sampling
        capture_frequency = cfg["capture_frequency"]
        simulation_frequency = cfg["fps"]
        delta_tick = int(simulation_frequency / capture_frequency)
        assert delta_tick > 0, "please reduce capture_frequency"

        # Initialize simulation variables
        frame_no = 0
        save_frame_no = 0
        call_exit = False
        
        # Get traffic actors for weather effects
        carla_world.weather.actors = carla_world.world.get_actors().filter('*vehicle*')
        
        # Compute camera intrinsics if capturing data
        for sensor in carla_world.ego_vehicle.sensors:
            if sensor.sensor_type == "RGBCamera" and cfg['capture_data']:
                _ = compute_K(sensor, out_dir)
        
        # Initialize asynchronous disk writer for efficient I/O
        writer = AsyncDiskWriter(
            num_workers=DISK_WRITER_WORKERS,
            max_queue_size=DISK_WRITER_QUEUE_SIZE
        )
        
        # ===== Main Simulation Loop =====
        while True:
            continue_flag = False
            
            # Advance simulation and get current frame ID
            frame_id = carla_world.tick()
            
            # Update weather conditions if enabled
            if cfg["dynamic_weather"]:
                carla_world.weather.tick(WEATHER_TICK_FACTOR * delta_tick)
                carla_world.world.set_weather(carla_world.weather.weather)

            # Get vehicle velocity once per frame (cache to avoid repeated calls)
            velocity = carla_world.ego_vehicle.ego_vehicle.get_velocity()
            velocity_magnitude = velocity.length()
            
            # Determine if this frame should be captured (based on frame skipping and velocity)
            do_capture = (
                frame_no % delta_tick == 0
                and cfg["capture_data"]
                and velocity_magnitude > MIN_VELOCITY_THRESHOLD
            )

            # Process all sensors and capture data if needed
            for sensor in carla_world.ego_vehicle.sensors:
                if sensor.sensor_type == "RGBCamera":
                    # Retrieve camera data
                    transform = sensor.get_transform()
                    rgb, depth, sem_seg, bb_2d = sensor.retrive_data(
                        frame_id, SENSOR_DATA_TIMEOUT
                    )
                    
                    # Queue camera data for async disk write
                    if do_capture:
                        capture_data_async(
                            writer=writer,
                            frame_no=save_frame_no,
                            out_dir=out_dir,
                            sensor_name=sensor.sensor_name,
                            rgb=rgb,
                            depth=depth,
                            semantic_mask=sem_seg,
                            bb_2d=bb_2d,
                            transform=transform
                        )

                elif sensor.sensor_type == "LiDAR":
                    # Retrieve LiDAR data
                    _, bbs, pcd = sensor.retrive_data(frame_id, SENSOR_DATA_TIMEOUT)
                    
                    # Skip frame if no bounding boxes detected
                    if do_capture:
                        if len(bbs) == 0:
                            continue_flag = True
                            break
                        
                        # Queue LiDAR data for async disk write
                        capture_data_async(
                            writer=writer,
                            frame_no=save_frame_no,
                            out_dir=out_dir,
                            sensor_name=sensor.sensor_name,
                            lidar_pc=pcd,
                            bbs=bbs,
                        )

            # Skip to next frame if no valid detections
            if continue_flag:
                continue

            # Increment save counter only when data is actually captured
            if do_capture or velocity_magnitude<=MIN_VELOCITY_THRESHOLD:
                save_frame_no += 1
                
            frame_no += 1

            # Handle visualization and user input
            if cfg["sensor_preview"]:
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        call_exit = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_q:
                            call_exit = True
                            break

                if call_exit:
                    pygame.display.quit()
                    for sensor in carla_world.ego_vehicle.sensors:
                        if sensor.sensor_type == "LiDAR" and sensor.vis:
                            sensor.vis.destroy_window()
                    break
    
    except Exception as e:
        print(f"Exception occurred during simulation: {e}")
        
    finally:
        # Clean up resources
        print("Destroying sensors...")
        for sensor in carla_world.ego_vehicle.sensors:
            sensor.destroy()

        print("Destroying actors...")
        carla_world.destroy_actors()
        carla_world.restore()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CARLA Data Collection and Simulation Script"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration file (config.yaml)"
    )
    args = parser.parse_args()
    main(args.config_path)
