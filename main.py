import yaml
import cv2
import glob
import os
import sys
import math

sys.path.append("src/")
try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

from src.pygame_display import DisplayManager
from src.world import CarlaWorld
from src.ego_vehicle import EgoVehicle

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass
import carla


def main():
    """
    Main function
    """
    try:
        with open("cfg\\vehicle_cfg.yaml", "r") as f:
            vehicle_cfg = yaml.safe_load(f)

        with open("cfg\\config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        carla_world = CarlaWorld(cfg)

        bp_lib = carla_world.world.get_blueprint_library()
        ego_vehicle = EgoVehicle(bp_lib, vehicle_cfg)
        ego_vehicle.spwan_ego_vehicle(carla_world.world)
        display_man = None
        if cfg["sensor_preview"]:
            grid_size = [math.ceil(ego_vehicle.num_cameras / 3), 3]
            display_man = DisplayManager(grid_size, window_size=[1280, 720])

        ego_vehicle.sensor_setup(carla_world.world, display_man, enable_lidar_vis=True)

        carla_world.spawn_actors()
        ego_vehicle.ego_vehicle.set_autopilot(True)

        carla_world.set_synchronous()

        out_dir = cfg["out_dir"]
        capture_frequency = cfg["capture_frequency"]
        simulation_frequency = cfg["fps"]
        delta_tick = int(simulation_frequency / capture_frequency)
        assert delta_tick > 0, "please reduce capture_frequency"

        call_exit = False

        # Main loop
        while True:
            frame_id = carla_world.tick()

            # Visualization
            if display_man:
                for sensor in ego_vehicle.sensors:
                    if sensor.sensor_type == "RGBCamera":
                        _, _, _ = sensor.retrive_data(frame_id, 2.0)

                    if sensor.sensor_type == "LiDAR":
                        _ = sensor.retrive_data(frame_id, 2.0)

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
                for sensor in ego_vehicle.sensors:
                    if sensor.sensor_type == "LiDAR" and sensor.vis:
                        sensor.vis.destroy_window()

                break

    except Exception as e:
        print(f"Exception {e}")

    finally:
        print("destroying actors")
        carla_world.restore()
        carla_world.destroy_actors()


if __name__ == "__main__":
    main()
