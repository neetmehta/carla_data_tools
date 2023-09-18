import glob
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time

import yaml
import queue

class EgoVehicle:
    
    def __init__(self, bp_lib, vehicle_cfg) -> None:
        
        self.vehicle_cfg = vehicle_cfg
        vehicle = self.vehicle_cfg['vehicle']
        self.blueprint_lib = bp_lib
        self.vehicles_bp = self.blueprint_lib.find(f'vehicle.{vehicle}')
        
        
    def spwan_ego_vehicle(self, world):
        spawn_points = world.get_map().get_spawn_points()
        self.ego_vehicle = world.try_spawn_actor(self.vehicles_bp, random.choice(spawn_points))
        self.spectator = world.get_spectator() 
        transform = carla.Transform(self.ego_vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),self.ego_vehicle.get_transform().rotation) 
        self.spectator.set_transform(transform)
        
    def sensor_setup(self, world):
        # Set initial camera translation
        sensors = self.vehicle_cfg['sensors']
        self.sensors = {}
        self.sensors_queues = {}
        
        for sensor in sensors:
            if sensor['sensor_type'] == 'rgb_camera':
                cam_location = carla.Location(*sensor['translation'])
                cam_rotation = carla.Rotation(*sensor['rotation'])
                camera_init_trans = carla.Transform(cam_location, cam_rotation)
                camera_bp = self.blueprint_lib.find('sensor.camera.rgb')
                camera_bp.set_attribute('image_size_x', str(sensor['image_size_x']))
                camera_bp.set_attribute('image_size_y', str(sensor['image_size_y']))
                camera_bp.set_attribute('fov', str(sensor['fov']))
                rgb_camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
                self.sensors[sensor['sensor_name']] = rgb_camera
                self.sensors_queues[sensor['sensor_name']] = queue.Queue()
                time.sleep(2.0)
                self.spectator.set_transform(rgb_camera.get_transform())
                
            if sensor['sensor_type'] == 'depth_camera':
                cam_location = carla.Location(*sensor['translation'])
                cam_rotation = carla.Rotation(*sensor['rotation'])
                camera_init_trans = carla.Transform(cam_location, cam_rotation)
                camera_bp = self.blueprint_lib.find('sensor.camera.depth')
                camera_bp.set_attribute('image_size_x', str(sensor['image_size_x']))
                camera_bp.set_attribute('image_size_y', str(sensor['image_size_y']))
                camera_bp.set_attribute('fov', str(sensor['fov']))
                depth_camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
                self.sensors[sensor['sensor_name']] = depth_camera
                self.sensors_queues[sensor['sensor_name']] = queue.Queue()
                time.sleep(2.0)
                self.spectator.set_transform(depth_camera.get_transform())
                
            if sensor['sensor_type'] == 'sem_seg_camera':
                cam_location = carla.Location(*sensor['translation'])
                cam_rotation = carla.Rotation(*sensor['rotation'])
                camera_init_trans = carla.Transform(cam_location, cam_rotation)
                camera_bp = self.blueprint_lib.find('sensor.camera.semantic_segmentation')
                camera_bp.set_attribute('image_size_x', str(sensor['image_size_x']))
                camera_bp.set_attribute('image_size_y', str(sensor['image_size_y']))
                camera_bp.set_attribute('fov', str(sensor['fov']))
                sem_seg_camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
                self.sensors[sensor['sensor_name']] = sem_seg_camera
                self.sensors_queues[sensor['sensor_name']] = queue.Queue()
                time.sleep(2.0)
                self.spectator.set_transform(sem_seg_camera.get_transform())
                
            if sensor['sensor_type'] == 'lidar':
                lidar_location = carla.Location(*sensor['translation'])
                lidar_rotation = carla.Rotation(*sensor['rotation'])
                lidar_init_trans = carla.Transform(lidar_location, lidar_rotation)
                lidar_bp = self.blueprint_lib.find('sensor.lidar.ray_cast')
                lidar_bp.set_attribute('channels',str(64))
                lidar_bp.set_attribute('points_per_second',str(100000))
                lidar_bp.set_attribute('rotation_frequency',str(1))
                lidar_bp.set_attribute('range',str(50))
                lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=self.ego_vehicle)
                self.sensors[sensor['sensor_name']] = lidar
                self.sensors_queues[sensor['sensor_name']] = queue.Queue()
