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
            if sensor['sensor_type'] == 'camera':
                cam_location = carla.Location(*sensor['translation'])
                cam_rotation = carla.Rotation(*sensor['rotation'])
                camera_init_trans = carla.Transform(cam_location, cam_rotation)
                self.camera_bp = self.blueprint_lib.find('sensor.camera.rgb')
                self.rgb_camera = world.spawn_actor(self.camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
                self.sensors[sensor['sensor_name']] = self.rgb_camera
                self.sensors_queues[sensor['sensor_name']] = queue.Queue()
                time.sleep(2.0)
                self.spectator.set_transform(self.rgb_camera.get_transform())
                
            if sensor['sensor_type'] == 'depth_camera':
                cam_location = carla.Location(*sensor['translation'])
                cam_rotation = carla.Rotation(*sensor['rotation'])
                camera_init_trans = carla.Transform(cam_location, cam_rotation)
                self.camera_bp = self.blueprint_lib.find('sensor.camera.rgb')
                self.rgb_camera = world.spawn_actor(self.camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
                self.sensors[sensor['sensor_name']] = self.rgb_camera
                self.sensors_queues[sensor['sensor_name']] = queue.Queue()
                time.sleep(2.0)
                self.spectator.set_transform(self.rgb_camera.get_transform())