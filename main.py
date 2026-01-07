import yaml
import glob
import os
import sys
import math
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


def main():
    """
    Main function
    """
    try:
        with open("cfg/kitti_config.yaml", "r") as f:
            vehicle_cfg = yaml.safe_load(f)

        with open("cfg/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        # os.makedirs(os.path.join(cfg["out_dir"], cfg["map"]), exist_ok=True)
        if not cfg["dynamic_weather"]:
            out_dir = os.path.join(cfg["out_dir"], f'run_{cfg["map"]}_{cfg["weather"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            
        else: 
            out_dir = os.path.join(cfg["out_dir"], f'run_{cfg["map"]}_dynamic_weather_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(out_dir, exist_ok=True)
        carla_world = CarlaWorldManager(cfg=cfg, vehicle_cfg=vehicle_cfg)

        carla_world.spawn_ego_vehicle()
        display_man = None
        if cfg["sensor_preview"]:
            grid_size = [math.ceil(carla_world.ego_vehicle.num_cameras / 3), min(carla_world.ego_vehicle.num_cameras, 3)]
            display_man = DisplayManager(grid_size, window_size=[1280, 720])

        carla_world.ego_vehicle.sensor_setup(
            carla_world.world, display_man, enable_lidar_vis=cfg["sensor_preview"]
        )
        carla_world.spawn_actors()
        carla_world.ego_vehicle.ego_vehicle.set_autopilot(True)
        carla_world.set_synchronous()

        capture_frequency = cfg["capture_frequency"]
        simulation_frequency = cfg["fps"]
        delta_tick = int(simulation_frequency / capture_frequency)
        assert delta_tick > 0, "please reduce capture_frequency"
        frame_no = 0
        save_frame_no = 0
        call_exit = False
        rgb, depth, sem_seg, _ = None, None, None, None
        carla_world.weather.actors = carla_world.world.get_actors().filter('*vehicle*')
        for sensor in carla_world.ego_vehicle.sensors:
            if sensor.sensor_type == "RGBCamera" and cfg['capture_data']:
                _ = compute_K(sensor, out_dir)
                
        writer = AsyncDiskWriter(num_workers=4, max_queue_size=200)
        # Main loop
        while True:
            continue_flag = False
            frame_id = carla_world.tick()
            if cfg["dynamic_weather"]:
                carla_world.weather.tick(1.0*delta_tick)
                carla_world.world.set_weather(carla_world.weather.weather)
                sys.stdout.write('\r' + str(carla_world.weather) + 12 * ' ')
                sys.stdout.flush()

            # Data Capture
            velocity = carla_world.ego_vehicle.ego_vehicle.get_velocity()
            do_capture = (
                        frame_no % delta_tick == 0
                        and cfg["capture_data"]
                        and velocity.length() > 0.001
                    )

            for sensor in carla_world.ego_vehicle.sensors:
                if sensor.sensor_type == "RGBCamera":
                    rgb, depth, sem_seg, bb_2d = sensor.retrive_data(frame_id, 2.0)
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
                        )
                        # print(f"Saved Camera Frame no {frame_no} for {sensor.sensor_name}")

                if sensor.sensor_type == "LiDAR":
                    _, bbs, pcd = sensor.retrive_data(frame_id, 2.0)
                    if do_capture:
                        if len(bbs) == 0:
                            continue_flag = True
                            break
                        capture_data_async(
                            writer=writer,
                            frame_no=save_frame_no,
                            out_dir=out_dir,
                            sensor_name=sensor.sensor_name,
                            lidar_pc=pcd,
                            bbs=bbs,
                        )
                        # print(f"Saved Lidar Frame no {frame_no} for {sensor.sensor_name}")
                if do_capture or velocity.length() <= 0.001:
                    # print(f"Captured data for frame no {frame_no}")
                    save_frame_no += 1
                    

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
                    for sensor in carla_world.ego_vehicle.sensors:
                        if sensor.sensor_type == "LiDAR" and sensor.vis:
                            sensor.vis.destroy_window()

                    break
    
    except Exception as e:
        print(f"Exception {e}") 
        
    finally:
        print("destroying sensors")
        for sensor in carla_world.ego_vehicle.sensors:
            sensor.destroy()

        print("destroying actors")
        carla_world.destroy_actors()
        carla_world.restore()


if __name__ == "__main__":
    main()
