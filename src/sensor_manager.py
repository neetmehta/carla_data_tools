import glob
import os
import sys
import matplotlib.pyplot as plt

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
import argparse
import random
import time
import numpy as np
import cv2
import open3d as o3d

from scipy.spatial.transform import Rotation
from queue import Queue

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

from utils import (
    process_rgb_image,
    process_depth_image,
    process_sem_seg_image,
    process_inst_seg_image,
    process_point_cloud,
    add_open3d_axis,
    is_empty,
    build_projection_matrix,
    get_image_point,
)

from bounding_box import ClientSideBoundingBoxes
from collections import namedtuple

BoundingBox = namedtuple("BoundingBox", ["center", "extent", "yaw"])

lines = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]

edges = [
    [0, 1],
    [1, 3],
    [3, 2],
    [2, 0],
    [0, 4],
    [4, 5],
    [5, 1],
    [5, 7],
    [7, 6],
    [6, 4],
    [6, 2],
    [7, 3],
]

SEMANTIC_MAP = {0: ('unlabelled', (0,0,0)), 1: ('road', (128,64,0)),2: ('sidewalk', (244,35,232)),
                3: ('building', (70,70,70)), 4: ('wall', (102,102,156)), 5: ('fence', (190,153,153)),
                6: ('pole', (153,153,153)), 7: ('traffic light', (250,170,30)), 
                8: ('traffic sign', (220,220,0)), 9: ('vegetation', (107,142,35)),
                10: ('terrain', (152,251,152)), 11: ('sky', (70,130,180)), 
                12: ('pedestrian', (220,20,60)), 13: ('rider', (255,0,0)), 
                14: ('car', (0,0,142)), 15: ('truck', (0,0,70)), 16: ('bus', (0,60,100)), 
                17: ('train', (0,80,100)), 18: ('motorcycle', (0,0,230)), 
                19: ('bicycle', (119,11,32)), 20: ('static', (110,190,160)), 
                21: ('dynamic', (170,120,50)), 22: ('other', (55,90,80)), 
                23: ('water', (45,60,150)), 24: ('road line', (157,234,50)), 
                25: ('ground', (81,0,81)), 26: ('bridge', (150,100,100)), 
                27: ('rail track', (230,150,140)), 28: ('guard rail', (180,165,180))}

class SensorBase:
    def __init__(self, world, ego_vehicle, sensor_cfg) -> None:
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.sensor_cfg = sensor_cfg
        self.sensor_type = sensor_cfg["sensor_type"]
        self.sensor_name = sensor_cfg["sensor_name"]
        sensor_location = carla.Location(*sensor_cfg["translation"])
        sensor_rotation = carla.Rotation(*sensor_cfg["rotation"])
        self.transform = carla.Transform(sensor_location, sensor_rotation)
        self.queue = Queue()
        self.processing_func = None
        self.vehicles = None

    def init_sensor(self):
        raise NotImplementedError

    def retrive_data(self, frame_id, timeout):
        while True:
            data = self.queue.get(timeout=timeout)
            if data.frame == frame_id:
                return self.processing_func(data)


class CameraSensor(SensorBase):
    def __init__(
        self,
        world,
        ego_vehicle,
        sensor_cfg,
        bp_lib,
        display_man=None,
        display_pos=None,
        depth=True,
        sem_seg=True,
        inst_seg=True
    ) -> None:
        super().__init__(world, ego_vehicle, sensor_cfg)
        self.depth = depth
        self.sem_seg = sem_seg
        self.inst_seg = inst_seg
        self.rgb_camera = None
        self.depth_camera = None
        self.sem_seg_camera = None
        self.depth_queue = None
        self.sem_seg_queue = None
        self.rgb_surface = None
        self.depth_surface = None
        self.sem_seg_surface = None

        self.display_man = display_man
        self.display_pos = [
            int(display_pos // 3),
            int(display_pos - (display_pos // 3) * 3),
        ]
        self.processing_func = process_rgb_image
        self.init_sensor(bp_lib)

    def init_sensor(self, bp_lib):
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.sensor_cfg["image_size_x"]))
        camera_bp.set_attribute("image_size_y", str(self.sensor_cfg["image_size_y"]))
        camera_bp.set_attribute("fov", str(self.sensor_cfg["fov"]))
        self.rgb_camera = self.world.spawn_actor(
            camera_bp, self.transform, attach_to=self.ego_vehicle
        )
        self.rgb_camera.listen(self.queue.put)

        self.cam_intrinsics = build_projection_matrix(
            self.sensor_cfg["image_size_x"],
            self.sensor_cfg["image_size_y"],
            self.sensor_cfg["fov"],
        )

        if self.depth:
            depth_camera_bp = bp_lib.find("sensor.camera.depth")
            depth_camera_bp.set_attribute(
                "image_size_x", str(self.sensor_cfg["image_size_x"])
            )
            depth_camera_bp.set_attribute(
                "image_size_y", str(self.sensor_cfg["image_size_y"])
            )
            depth_camera_bp.set_attribute("fov", str(self.sensor_cfg["fov"]))
            self.depth_camera = self.world.spawn_actor(
                depth_camera_bp, self.transform, attach_to=self.ego_vehicle
            )
            self.depth_queue = Queue()
            self.depth_camera.listen(self.depth_queue.put)

        if self.sem_seg:
            sem_seg_camera_bp = bp_lib.find("sensor.camera.semantic_segmentation")
            sem_seg_camera_bp.set_attribute(
                "image_size_x", str(self.sensor_cfg["image_size_x"])
            )
            sem_seg_camera_bp.set_attribute(
                "image_size_y", str(self.sensor_cfg["image_size_y"])
            )
            sem_seg_camera_bp.set_attribute("fov", str(self.sensor_cfg["fov"]))
            self.sem_seg_camera = self.world.spawn_actor(
                sem_seg_camera_bp, self.transform, attach_to=self.ego_vehicle
            )
            self.sem_seg_queue = Queue()
            self.sem_seg_camera.listen(self.sem_seg_queue.put)
        
        if self.inst_seg:
            inst_seg_camera_bp = bp_lib.find("sensor.camera.instance_segmentation")
            inst_seg_camera_bp.set_attribute(
                "image_size_x", str(self.sensor_cfg["image_size_x"])
            )
            inst_seg_camera_bp.set_attribute(
                "image_size_y", str(self.sensor_cfg["image_size_y"])
            )
            inst_seg_camera_bp.set_attribute("fov", str(self.sensor_cfg["fov"]))
            self.inst_seg_camera = self.world.spawn_actor(
                inst_seg_camera_bp, self.transform, attach_to=self.ego_vehicle
            )
            self.inst_seg_queue = Queue()
            self.inst_seg_camera.listen(self.inst_seg_queue.put)


    def retrive_data(self, frame_id, timeout):
        rgb_data = super().retrive_data(frame_id, timeout)
        depth_data = None
        sem_seg_data = None
        if self.depth:
            while True:
                depth_data = self.depth_queue.get(timeout=timeout)
                if depth_data.frame == frame_id:
                    depth_data = process_depth_image(depth_data)
                    break
        if self.sem_seg:
            while True:
                sem_seg_data = self.sem_seg_queue.get(timeout=timeout)
                if sem_seg_data.frame == frame_id:
                    sem_seg_data = process_sem_seg_image(sem_seg_data)
                    break
        if self.inst_seg:
            while True:
                inst_seg_data = self.inst_seg_queue.get(timeout=timeout)
                if inst_seg_data.frame == frame_id:
                    inst_seg_data = process_inst_seg_image(inst_seg_data)
                    break
        bb_2d = self.get_bbox_2d(inst_seg_data)
        if self.display_man is not None:           

            self.rgb_surface = pygame.surfarray.make_surface(
                rgb_data.swapaxes(0, 1)
            )
            self.rgb_surface = self.draw_cam_bbs(rgb_data, bb_2d, self.rgb_surface)
            self.render()

        return rgb_data, depth_data, sem_seg_data, bb_2d

    def get_bbox_2d(self, inst_seg):
        semantic_labels, actor_ids = inst_seg
        dynamic_class = [12,13,14,15,16,17,18,19]
        boxes = []

        for semantic_class in dynamic_class:
            mask = semantic_labels == semantic_class
        
            unique_actors = np.unique(actor_ids[mask])
            for unique_actor in unique_actors:
                if unique_actor == self.ego_vehicle.id:
                    continue
                actor_mask = actor_ids == unique_actor
                ys, xs = np.where(actor_mask)
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                bbox_area = ((xmax-xmin)*(ymax-ymin))
                if bbox_area>300:
                    boxes.append({'actor_id': int(unique_actor),
                        'semantic_label': semantic_class,
                        'bbox_2d': (int(xmin), int(ymin), int(xmax), int(ymax))})
                    
        return boxes
        
    def draw_cam_bbs(self, img, boxes, surface):
        
        rgb_img = img[:, :, :3][:, :, ::-1] 
        frame_surface = pygame.surfarray.make_surface(np.transpose(rgb_img[..., 0:3], (1,0,2)))
        surface.blit(frame_surface, (0, 0))

        font = pygame.font.SysFont("Arial", 18)

        for bbox in boxes:
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(v) for v in bbox['bbox_2d']]
                label = SEMANTIC_MAP[bbox['semantic_label']][0]
                color = SEMANTIC_MAP[bbox['semantic_label']][1]
                pygame.draw.rect(surface, color, pygame.Rect(xmin, ymin, xmax-xmin, ymax-ymin), 2)
                text_surface = font.render(label, True, (255,255,255), color) 
                text_rect = text_surface.get_rect(topleft=(xmin, ymin-20))
                surface.blit(text_surface, text_rect)

        return surface


    def render(self):
        if self.rgb_surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.rgb_surface, offset)

    def destroy(self):
        # self.rgb_camera.stop()
        self.rgb_camera.destroy()
        if self.depth:
            # self.depth_camera.stop()
            self.depth_camera.destroy()

        if self.sem_seg:
            # self.sem_seg_camera.stop()
            self.sem_seg_camera.destroy()
            
        if self.inst_seg:
            # self.inst_seg_camera.stop()
            self.inst_seg_camera.destroy()
            
    def get_transform(self):
        return self.rgb_camera.get_transform()


class LidarSensor(SensorBase):
    def __init__(
        self, world, ego_vehicle, sensor_cfg, bp_lib, enable_visualization=True
    ) -> None:
        super().__init__(world, ego_vehicle, sensor_cfg)
        self.lidar = None
        self.vis = None
        if enable_visualization:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=sensor_cfg["sensor_name"],
                width=960,
                height=540,
                left=480,
                top=270,
            )
            self.vis.get_render_option().background_color = [1.0, 1.0, 1.0]
            self.vis.get_render_option().point_size = 1
            self.vis.get_render_option().show_coordinate_frame = True
            add_open3d_axis(self.vis)
            self.empty_line_set = o3d.geometry.LineSet()
            self.empty_points = self.empty_line_set.points

        self.processing_func = process_point_cloud
        self.init_sensor(bp_lib)
        self.static_bboxes = self.world.get_level_bbs(carla.CityObjectLabel.Car)
        self.frame = 0
        self.pcd = o3d.geometry.PointCloud()
        self.pcd_save = o3d.t.geometry.PointCloud()
        self.line_sets = []

    def init_sensor(self, bp_lib):
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(self.sensor_cfg["channels"]))
        lidar_bp.set_attribute(
            "points_per_second", str(self.sensor_cfg["points_per_second"])
        )
        lidar_bp.set_attribute(
            "rotation_frequency", str(self.sensor_cfg["rotation_frequency"])
        )
        lidar_bp.set_attribute("noise_stddev", str(self.sensor_cfg["noise_stddev"]))
        lidar_bp.set_attribute("upper_fov", str(self.sensor_cfg["upper_fov"]))
        lidar_bp.set_attribute("lower_fov", str(self.sensor_cfg["lower_fov"]))
        lidar_bp.set_attribute("range", str(self.sensor_cfg["range"]))
        lidar_bp.set_attribute("sensor_tick", str(self.sensor_cfg["sensor_tick"]))
        lidar_bp.set_attribute(
            "dropoff_general_rate", str(self.sensor_cfg["dropoff_general_rate"])
        )
        lidar_bp.set_attribute(
            "dropoff_intensity_limit", str(self.sensor_cfg["dropoff_intensity_limit"])
        )
        lidar_bp.set_attribute(
            "dropoff_zero_intensity", str(self.sensor_cfg["dropoff_zero_intensity"])
        )

        self.lidar = self.world.spawn_actor(
            lidar_bp, self.transform, attach_to=self.ego_vehicle
        )
        self.lidar.listen(self.queue.put)

    def retrive_data(self, frame_id, timeout):
        points, colors, intensity = super().retrive_data(frame_id, timeout)
        intensity = [[i] for i in intensity]

        self.pcd_save.point["positions"] = o3d.core.Tensor(points)
        self.pcd_save.point["intensities"] = o3d.core.Tensor(intensity)

        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        if self.vehicles is None:
            self.vehicles = list(self.world.get_actors().filter("vehicle.*"))
        bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(
            self.ego_vehicle,
            self.vehicles,
            self.lidar,
            additional_bb=self.static_bboxes,
        )
        bbs = []

        for box in bounding_boxes:
            box[0, :] *= -1

        if self.vis:
            if self.frame == 2:
                self.vis.add_geometry(self.pcd)

                for box in bounding_boxes:
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(np.asarray(box.T))
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    self.vis.add_geometry(line_set)
                    self.line_sets.append(line_set)

            if len(self.line_sets) != 0:
                for i, box in enumerate(bounding_boxes):
                    if is_empty(
                        self.pcd,
                        box,
                        threshold=self.sensor_cfg["vehicle_detection_threshold"],
                    ):
                        self.line_sets[i].points = self.empty_points
                        self.vis.update_geometry(self.line_sets[i])
                        continue
                    self.line_sets[i].points = o3d.utility.Vector3dVector(
                        np.asarray(box.T)
                    )
                    self.vis.update_geometry(self.line_sets[i])
                    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                        o3d.utility.Vector3dVector(np.array(box.T))
                    )
                    r = Rotation.from_matrix(obb.R.copy())
                    _, _, yaw = r.as_euler("xyz", degrees=True)
                    bb = BoundingBox(
                        obb.center.copy(), obb.extent.copy(), np.deg2rad(yaw)
                    )
                    bbs.append(bb)

            self.vis.update_geometry(self.pcd)

            for line_set in self.line_sets:
                self.vis.update_geometry(line_set)

            self.vis.poll_events()
            self.vis.update_renderer()

            time.sleep(0.005)
            self.frame += 1

        return self.pcd, bbs, self.pcd_save

    def destroy(self):
        # self.lidar.stop()
        self.lidar.destroy()
