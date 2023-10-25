import yaml
import glob
import os
import sys
import math
import time

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
from src.utils import capture_data

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
    with open("cfg\\vehicle_cfg.yaml", "r") as f:
        vehicle_cfg = yaml.safe_load(f)

    with open("cfg\\config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.join(cfg["out_dir"], cfg["map"]), exist_ok=True)
    out_dir = os.path.join(cfg["out_dir"], cfg["map"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    carla_world = CarlaWorld(cfg)

    bp_lib = carla_world.world.get_blueprint_library()
    ego_vehicle = EgoVehicle(bp_lib, vehicle_cfg)
    ego_vehicle.spwan_ego_vehicle(carla_world.world)
    display_man = None
    if cfg["sensor_preview"]:
        grid_size = [math.ceil(ego_vehicle.num_cameras / 3), 3]
        display_man = DisplayManager(grid_size, window_size=[1280, 720])

    ego_vehicle.sensor_setup(
        carla_world.world, display_man, enable_lidar_vis=cfg["sensor_preview"]
    )
    carla_world.spawn_actors()
    ego_vehicle.ego_vehicle.set_autopilot(True)

    carla_world.set_synchronous()

    capture_frequency = cfg["capture_frequency"]
    simulation_frequency = cfg["fps"]
    delta_tick = int(simulation_frequency / capture_frequency)
    assert delta_tick > 0, "please reduce capture_frequency"
    frame_no = 0
    call_exit = False
    rgb, depth, sem_seg, point_cloud = None, None, None, None
    # Main loop
    while True:
        continue_flag = False
        frame_id = carla_world.tick()

        # Data Capture
        velocity = ego_vehicle.ego_vehicle.get_velocity()

        for sensor in ego_vehicle.sensors:
            if sensor.sensor_type == "RGBCamera":
                rgb, depth, sem_seg, bb_2d = sensor.retrive_data(frame_id, 2.0)
                if (
                    frame_no % delta_tick == 0
                    and cfg["capture_data"]
                    and velocity.length() > 0.001
                ):
                    capture_data(
                        frame_no=frame_no,
                        out_dir=out_dir,
                        sensor_name=sensor.sensor_name,
                        rgb=rgb,
                        depth=depth,
                        semantic_mask=sem_seg,
                        bb_2d=bb_2d,
                    )
                    # print(f"Saved Camera Frame no {frame_no} for {sensor.sensor_name}")

            if sensor.sensor_type == "LiDAR":
                point_cloud, bbs, pcd = sensor.retrive_data(frame_id, 2.0)
                if (
                    frame_no % delta_tick == 0
                    and cfg["capture_data"]
                    and velocity.length() > 0.001
                ):
                    if len(bbs) == 0:
                        continue_flag = True
                        break
                    capture_data(
                        frame_no=frame_no,
                        out_dir=out_dir,
                        sensor_name=sensor.sensor_name,
                        lidar_pc=pcd,
                        bbs=bbs,
                    )
                    # print(f"Saved Lidar Frame no {frame_no} for {sensor.sensor_name}")

        if continue_flag:
            continue

        frame_no += 1

        # Visualization
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
                for sensor in ego_vehicle.sensors:
                    if sensor.sensor_type == "LiDAR" and sensor.vis:
                        sensor.vis.destroy_window()

                break

    for sensor in ego_vehicle.sensors:
        sensor.destroy()

    print("destroying actors")
    carla_world.destroy_actors()
    carla_world.restore()


if __name__ == "__main__":
    main()
