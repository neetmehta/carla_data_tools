from world import CarlaWorld
from ego_vehicle import EgoVehicle
import yaml
import cv2
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
import open3d as o3d
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

from utils import process_rgb_image, retrive_data, process_depth_image, process_sem_seg_image, process_point_cloud, add_open3d_axis
from bounding_box import ClientSideBoundingBoxes

lines = [[0, 1], [1, 2], [2, 3], [3, 0],
         [4, 5], [5, 6], [6, 7], [7, 4],
         [0, 4], [1, 5], [2, 6], [3, 7]]

def main():
    with open('cfg\\vehicle_cfg.yaml', 'r') as f:
        vehicle_cfg = yaml.safe_load(f) 
    
    with open('cfg\\config.yaml', 'r') as f:
        cfg = yaml.safe_load(f) 
        
    carla_world = CarlaWorld(cfg)
    carla_world.spawn_actors()
    bp_lib = carla_world.world.get_blueprint_library()
    ego_vehicle = EgoVehicle(bp_lib, vehicle_cfg)
    ego_vehicle.spwan_ego_vehicle(carla_world.world)
    ego_vehicle.sensor_setup(carla_world.world)
    ego_vehicle.ego_vehicle.set_autopilot(True)
    for sensor_name, sensor in ego_vehicle.sensors.items():
        sensor.listen(ego_vehicle.sensors_queues[sensor_name].put)
    carla_world.set_synchronous()
    point_list = o3d.geometry.PointCloud()

    # Open3D visualiser for LIDAR and RADAR
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Carla Lidar',
        width=960,
        height=540,
        left=480,
        top=270)
    vis.get_render_option().background_color = [1.0, 1.0, 1.0]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True
    add_open3d_axis(vis)
    frame = 0
    while True:
        data = {}
        frame_id = carla_world.tick()
        for sensor_name, sensor in ego_vehicle.sensors.items():
            data[sensor_name] = retrive_data(ego_vehicle.sensors_queues[sensor_name], frame_id, 2.0)
        
        rgb_array = process_rgb_image(data['rgb_camera1'])
        # sem_seg_array = process_sem_seg_image(data['sem_seg_camera1'])
        # depth_array = process_depth_image(data['depth_camera1'])
        process_point_cloud(data['lidar1'], point_list)
        vehicles = carla_world.world.get_actors().filter('vehicle.*')
        bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(ego_vehicle.ego_vehicle, vehicles, ego_vehicle.sensors['lidar1'])
        for box in bounding_boxes:
            box[0,:] *= -1

        if frame == 2:
            vis.add_geometry(point_list)
            line_sets = []
            for box in bounding_boxes:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(np.asarray(box.T))
                line_set.lines = o3d.utility.Vector2iVector(lines)
                vis.add_geometry(line_set)
                line_sets.append(line_set)
            
        for i, box in enumerate(bounding_boxes):
            line_sets[i].points = o3d.utility.Vector3dVector(np.asarray(box.T))
            vis.update_geometry(line_sets[i])
            
        vis.update_geometry(point_list)

        for line_set in line_sets:
            vis.update_geometry(line_set)
        
        vis.poll_events()
        vis.update_renderer()
        
        # # This can fix Open3D jittering issues:
        time.sleep(0.005)
        frame += 1
        cv2.imshow('RGB Camera', rgb_array)
        # cv2.imshow('Depth Camera', depth_array)
        # cv2.imshow('Semantic Camera', sem_seg_array)

        # Quit if user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            vis.destroy_window()
            
            break
        

    carla_world.restore()
    carla_world.destroy_actors()

    
if __name__ == "__main__":
    
    main()