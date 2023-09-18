from world import CarlaWorld
from ego_vehicle import EgoVehicle
import yaml
import cv2
import numpy as np

from utils import process_rgb_image, retrive_data

def main():
    with open('cfg\\vehicle_cfg.yaml', 'r') as f:
        vehicle_cfg = yaml.safe_load(f) 
        
    with open('cfg\\config.yaml', 'r') as f:
        cfg = yaml.safe_load(f) 
        
    carla_world = CarlaWorld(cfg)
    carla_world.spawn_actors(10)
    bp_lib = carla_world.world.get_blueprint_library()
    ego_vehicle = EgoVehicle(bp_lib, vehicle_cfg)
    ego_vehicle.spwan_ego_vehicle(carla_world.world)
    ego_vehicle.sensor_setup(carla_world.world)
    ego_vehicle.ego_vehicle.set_autopilot(True)
    for sensor_name, sensor in ego_vehicle.sensors.items():
        sensor.listen(ego_vehicle.sensors_queues[sensor_name].put)
    carla_world.set_synchronous()

    i=0
    while True:
        data = {}
        frame_id = carla_world.tick()
        for sensor_name, sensor in ego_vehicle.sensors.items():
            data[sensor_name] = retrive_data(ego_vehicle.sensors_queues[sensor_name], frame_id, 2.0)
        
        rgb_array = process_rgb_image(data['rgb_camera1'])
        # sem_seg_array = process_sem_seg_image(data['sem_seg_camera1'])
        depth_array = process_depth_image(data['depth_camera1'])
        cv2.imshow('RGB Camera', rgb_array)

        # Quit if user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            
            break

    carla_world.restore()
    carla_world.destroy_actors()
    
if __name__ == "__main__":
    
    main()