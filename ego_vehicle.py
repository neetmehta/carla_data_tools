import glob
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import process_rgb_image
from sensor_manager import CameraSensor, LidarSensor

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

import random
import time

import yaml
import queue


class EgoVehicle:
    def __init__(self, bp_lib, vehicle_cfg) -> None:
        self.vehicle_cfg = vehicle_cfg
        vehicle = self.vehicle_cfg["vehicle"]
        self.blueprint_lib = bp_lib
        self.vehicles_bp = self.blueprint_lib.find(f"vehicle.{vehicle}")
        self.num_cameras = 0
        for i in vehicle_cfg["sensors"]:
            if i['sensor_type'] == 'RGBCamera':
                self.num_cameras += 1

    def spwan_ego_vehicle(self, world):
        spawn_points = world.get_map().get_spawn_points()
        self.ego_vehicle = world.try_spawn_actor(
            self.vehicles_bp, random.choice(spawn_points)
        )
        self.spectator = world.get_spectator()
        transform = carla.Transform(
            self.ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
            self.ego_vehicle.get_transform().rotation,
        )
        self.spectator.set_transform(transform)

    def sensor_setup(self, world, display_man, enable_lidar_vis=False):
        # Set initial camera translation
        sensors = self.vehicle_cfg["sensors"]
        self.sensors = []
        
        cam_display_pos = 0
        for sensor_cfg in sensors:
            if sensor_cfg["sensor_type"] == "RGBCamera":
                camera = CameraSensor(world, self.ego_vehicle, sensor_cfg, self.blueprint_lib, display_man, cam_display_pos, sensor_cfg['depth'], sensor_cfg['sem_seg'])
                self.sensors.append(camera)
                cam_display_pos += 1
                
            if sensor_cfg["sensor_type"] == "LiDAR":
                lidar = LidarSensor(world, self.ego_vehicle, sensor_cfg, self.blueprint_lib, enable_lidar_vis)
                self.sensors.append(lidar)
            