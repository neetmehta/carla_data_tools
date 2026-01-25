"""
Sensor Management Module for CARLA

This module provides classes to manage RGB cameras, depth cameras, semantic segmentation,
instance segmentation, and LiDAR sensors in CARLA simulator. It handles sensor data
retrieval, processing, visualization, and bounding box detection.

Classes:
    SensorBase: Abstract base class for all sensors
    CameraSensor: Manages RGB, depth, and segmentation camera sensors
    LidarSensor: Manages LiDAR sensor with point cloud processing
"""

import carla
import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from queue import Queue
from collections import namedtuple

try:
    import pygame
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
)
from bounding_box import ClientSideBoundingBoxes

# Named tuple for bounding box representation
BoundingBox = namedtuple("BoundingBox", ["center", "extent", "yaw"])

# 3D bounding box wireframe edges
BOUNDING_BOX_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Front face
    [4, 5], [5, 6], [6, 7], [7, 4],  # Back face
    [0, 4], [1, 5], [2, 6], [3, 7],  # Connecting edges
]

# Semantic class label to color mapping (CARLA semantic segmentation)
SEMANTIC_MAP = {
    0: ("unlabelled", (0, 0, 0)),
    1: ("road", (128, 64, 0)),
    2: ("sidewalk", (244, 35, 232)),
    3: ("building", (70, 70, 70)),
    4: ("wall", (102, 102, 156)),
    5: ("fence", (190, 153, 153)),
    6: ("pole", (153, 153, 153)),
    7: ("traffic light", (250, 170, 30)),
    8: ("traffic sign", (220, 220, 0)),
    9: ("vegetation", (107, 142, 35)),
    10: ("terrain", (152, 251, 152)),
    11: ("sky", (70, 130, 180)),
    12: ("pedestrian", (220, 20, 60)),
    13: ("rider", (255, 0, 0)),
    14: ("car", (0, 0, 142)),
    15: ("truck", (0, 0, 70)),
    16: ("bus", (0, 60, 100)),
    17: ("train", (0, 80, 100)),
    18: ("motorcycle", (0, 0, 230)),
    19: ("bicycle", (119, 11, 32)),
    20: ("static", (110, 190, 160)),
    21: ("dynamic", (170, 120, 50)),
    22: ("other", (55, 90, 80)),
    23: ("water", (45, 60, 150)),
    24: ("road line", (157, 234, 50)),
    25: ("ground", (81, 0, 81)),
    26: ("bridge", (150, 100, 100)),
    27: ("rail track", (230, 150, 140)),
    28: ("guard rail", (180, 165, 180)),
}

# Dynamic object classes for bounding box detection
DYNAMIC_OBJECT_CLASSES = [12, 13, 14, 15, 16, 17, 18, 19]  # Pedestrian, rider, car, truck, bus, train, motorcycle, bicycle
MIN_BBOX_AREA = 300  # Minimum bounding box pixel area for detection

# LiDAR visualization parameters
OPEN3D_VIS_WIDTH = 960
OPEN3D_VIS_HEIGHT = 540
OPEN3D_VIS_LEFT = 480
OPEN3D_VIS_TOP = 270
OPEN3D_POINT_SIZE = 1
LIDAR_SLEEP_TIME = 0.005  # Sleep duration for visualization update


class SensorBase:
    """
    Abstract base class for all sensor types in CARLA.
    
    Provides common functionality for sensor initialization, data retrieval,
    and queue management.
    
    Attributes:
        world: CARLA world object
        ego_vehicle: CARLA ego vehicle actor
        sensor_cfg: Sensor configuration dictionary
        sensor_type: Type of sensor (e.g., "RGBCamera", "LiDAR")
        sensor_name: Name identifier for the sensor
        transform: Sensor's location and rotation relative to ego vehicle
        queue: Queue for receiving sensor data
        processing_func: Function to process raw sensor data
        vehicles: List of vehicles in the world
    """
    
    def __init__(self, world, ego_vehicle, sensor_cfg) -> None:
        """
        Initialize base sensor.
        
        Args:
            world: CARLA world instance
            ego_vehicle: CARLA ego vehicle actor
            sensor_cfg: Dictionary containing sensor configuration
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.sensor_cfg = sensor_cfg
        self.sensor_type = sensor_cfg["sensor_type"]
        self.sensor_name = sensor_cfg["sensor_name"]
        
        # Build sensor transform from config
        sensor_location = carla.Location(*sensor_cfg["translation"])
        sensor_rotation = carla.Rotation(*sensor_cfg["rotation"])
        self.transform = carla.Transform(sensor_location, sensor_rotation)
        
        self.queue = Queue()
        self.processing_func = None
        self.vehicles = None

    def init_sensor(self):
        """Initialize sensor (to be implemented by subclasses)."""
        raise NotImplementedError

    def retrive_data(self, frame_id, timeout):
        """
        Retrieve sensor data for a specific frame.
        
        Args:
            frame_id: Frame number to retrieve data for
            timeout: Timeout for waiting for data (seconds)
            
        Returns:
            Processed sensor data
        """
        while True:
            data = self.queue.get(timeout=timeout)
            if data.frame == frame_id:
                return self.processing_func(data)


class CameraSensor(SensorBase):
    """
    RGB Camera with optional depth, semantic segmentation, and instance segmentation.
    
    Manages multiple camera sensors attached to the ego vehicle and provides
    methods for retrieving and processing image data, detecting 2D bounding boxes,
    and rendering visualizations.
    
    Attributes:
        rgb_camera: RGB camera actor
        depth_camera: Depth camera actor (optional)
        sem_seg_camera: Semantic segmentation camera actor (optional)
        inst_seg_camera: Instance segmentation camera actor (optional)
        cam_intrinsics: Camera intrinsic matrix for projection
        display_man: Display manager for visualization
    """
    
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
        inst_seg=True,
    ) -> None:
        """
        Initialize camera sensor.
        
        Args:
            world: CARLA world instance
            ego_vehicle: CARLA ego vehicle actor
            sensor_cfg: Camera configuration dictionary
            bp_lib: CARLA blueprint library
            display_man: Display manager for rendering (optional)
            display_pos: Display position index (optional)
            depth: Enable depth camera (default: True)
            sem_seg: Enable semantic segmentation camera (default: True)
            inst_seg: Enable instance segmentation camera (default: True)
        """
        super().__init__(world, ego_vehicle, sensor_cfg)
        self.depth = depth
        self.sem_seg = sem_seg
        self.inst_seg = inst_seg
        self.rgb_camera = None
        self.depth_camera = None
        self.sem_seg_camera = None
        self.inst_seg_camera = None
        self.depth_queue = None
        self.sem_seg_queue = None
        self.inst_seg_queue = None
        self.cam_intrinsics = None

        self.display_man = display_man
        if display_pos is not None:
            self.display_pos = [
                int(display_pos // 3),
                int(display_pos - (display_pos // 3) * 3),
            ]
        
        self.processing_func = process_rgb_image
        self.init_sensor(bp_lib)

    def init_sensor(self, bp_lib):
        """
        Initialize all camera sensors and attach to ego vehicle.
        
        Args:
            bp_lib: CARLA blueprint library
        """
        # Initialize RGB camera
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.sensor_cfg["image_size_x"]))
        camera_bp.set_attribute("image_size_y", str(self.sensor_cfg["image_size_y"]))
        camera_bp.set_attribute("fov", str(self.sensor_cfg["fov"]))
        self.rgb_camera = self.world.spawn_actor(
            camera_bp, self.transform, attach_to=self.ego_vehicle
        )
        self.rgb_camera.listen(self.queue.put)

        # Compute camera intrinsic matrix
        self.cam_intrinsics = build_projection_matrix(
            self.sensor_cfg["image_size_x"],
            self.sensor_cfg["image_size_y"],
            self.sensor_cfg["fov"],
        )

        # Initialize depth camera if enabled
        if self.depth:
            self._init_depth_camera(bp_lib)

        # Initialize semantic segmentation camera if enabled
        if self.sem_seg:
            self._init_sem_seg_camera(bp_lib)

        # Initialize instance segmentation camera if enabled
        if self.inst_seg:
            self._init_inst_seg_camera(bp_lib)

    def _init_depth_camera(self, bp_lib):
        """Initialize depth camera sensor."""
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

    def _init_sem_seg_camera(self, bp_lib):
        """Initialize semantic segmentation camera sensor."""
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

    def _init_inst_seg_camera(self, bp_lib):
        """Initialize instance segmentation camera sensor."""
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
        """
        Retrieve and process all camera data for a specific frame.
        
        Args:
            frame_id: Frame number to retrieve data for
            timeout: Timeout for waiting for sensor data
            
        Returns:
            Tuple of (rgb_data, depth_data, sem_seg_data, bb_2d)
        """
        # Retrieve RGB data
        rgb_data = super().retrive_data(frame_id, timeout)
        
        # Retrieve depth data if enabled
        depth_data = self._retrieve_data_from_queue(
            self.depth_queue, frame_id, timeout, process_depth_image
        ) if self.depth else None
        
        # Retrieve semantic segmentation data if enabled
        sem_seg_data = self._retrieve_data_from_queue(
            self.sem_seg_queue, frame_id, timeout, process_sem_seg_image
        ) if self.sem_seg else None
        
        # Retrieve instance segmentation data if enabled
        inst_seg_data = self._retrieve_data_from_queue(
            self.inst_seg_queue, frame_id, timeout, process_inst_seg_image
        ) if self.inst_seg else None
        
        # Extract 2D bounding boxes from instance segmentation
        bb_2d = self.get_bbox_2d(inst_seg_data) if inst_seg_data is not None else []
        
        # Render visualization if enabled
        if self.display_man is not None:
            self._render_visualization(rgb_data, bb_2d)

        return rgb_data, depth_data, sem_seg_data, bb_2d

    def _retrieve_data_from_queue(self, data_queue, frame_id, timeout, process_func):
        """
        Helper method to retrieve and process data from queue.
        
        Args:
            data_queue: Queue to retrieve data from
            frame_id: Target frame ID
            timeout: Timeout for queue operation
            process_func: Function to process the data
            
        Returns:
            Processed data or None if queue is None
        """
        if data_queue is None:
            return None
            
        while True:
            data = data_queue.get(timeout=timeout)
            if data.frame == frame_id:
                return process_func(data)

    def get_bbox_2d(self, inst_seg):
        """
        Extract 2D bounding boxes from instance segmentation data.
        
        Detects bounding boxes for dynamic objects (pedestrians, vehicles, etc.)
        from instance segmentation masks.
        
        Args:
            inst_seg: Tuple of (semantic_labels, actor_ids) from instance seg camera
            
        Returns:
            List of bounding boxes with format:
            [{
                "actor_id": int,
                "semantic_label": int,
                "bbox_2d": (xmin, ymin, xmax, ymax)
            }]
        """
        semantic_labels, actor_ids = inst_seg
        boxes = []

        # Iterate through dynamic object classes
        for semantic_class in DYNAMIC_OBJECT_CLASSES:
            mask = semantic_labels == semantic_class
            unique_actors = np.unique(actor_ids[mask])
            
            for unique_actor in unique_actors:
                # Skip ego vehicle
                if unique_actor == self.ego_vehicle.id:
                    continue
                
                # Get pixel coordinates of actor
                actor_mask = actor_ids == unique_actor
                ys, xs = np.where(actor_mask)
                
                # Skip if no pixels found
                if len(xs) == 0 or len(ys) == 0:
                    continue
                
                # Calculate bounding box
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                bbox_area = (xmax - xmin) * (ymax - ymin)
                
                # Only include boxes above minimum area threshold
                if bbox_area > MIN_BBOX_AREA:
                    boxes.append({
                        "actor_id": int(unique_actor),
                        "semantic_label": semantic_class,
                        "bbox_2d": (int(xmin), int(ymin), int(xmax), int(ymax)),
                    })

        return boxes

    def _render_visualization(self, img, boxes):
        """
        Render RGB image with bounding boxes to display.
        
        Args:
            img: RGB image data
            boxes: List of bounding boxes to draw
        """
        rgb_img = img[:, :, :3][:, :, ::-1]
        frame_surface = pygame.surfarray.make_surface(
            np.transpose(rgb_img[..., 0:3], (1, 0, 2))
        )
        
        font = pygame.font.SysFont("Arial", 18)

        # Draw each bounding box
        for bbox in boxes:
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(v) for v in bbox["bbox_2d"]]
                label = SEMANTIC_MAP[bbox["semantic_label"]][0]
                color = SEMANTIC_MAP[bbox["semantic_label"]][1]
                
                # Draw rectangle
                pygame.draw.rect(
                    frame_surface, color,
                    pygame.Rect(xmin, ymin, xmax - xmin, ymax - ymin), 2
                )
                
                # Draw label text
                text_surface = font.render(label, True, (255, 255, 255), color)
                text_rect = text_surface.get_rect(topleft=(xmin, ymin - 20))
                frame_surface.blit(text_surface, text_rect)

        # Render to display
        self.render(frame_surface)

    def render(self, surface):
        """
        Render camera image to display window.
        
        Args:
            surface: Pygame surface containing the image
        """
        if surface is not None and self.display_man is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            resize = self.display_man.get_display_size()
            self.display_man.display.blit(
                pygame.transform.scale(surface, resize), offset
            )

    def destroy(self):
        """Destroy all camera sensors."""
        self.rgb_camera.destroy()
        if self.depth:
            self.depth_camera.destroy()
        if self.sem_seg:
            self.sem_seg_camera.destroy()
        if self.inst_seg:
            self.inst_seg_camera.destroy()

    def get_transform(self):
        """Get current transform of RGB camera."""
        return self.rgb_camera.get_transform()


class LidarSensor(SensorBase):
    """
    LiDAR sensor with point cloud processing and 3D visualization.
    
    Manages LiDAR data capture, point cloud processing, and 3D bounding box
    detection using Open3D visualization.
    
    Attributes:
        lidar: CARLA LiDAR sensor actor
        vis: Open3D visualizer for point cloud rendering
        pcd: Open3D point cloud object
        pcd_save: Open3D tensor point cloud for efficient storage
        line_sets: List of Open3D line sets for bounding boxes
        static_bboxes: Bounding boxes of static objects in the world
    """
    
    def __init__(
        self, world, ego_vehicle, sensor_cfg, bp_lib, enable_visualization=True
    ) -> None:
        """
        Initialize LiDAR sensor.
        
        Args:
            world: CARLA world instance
            ego_vehicle: CARLA ego vehicle actor
            sensor_cfg: LiDAR configuration dictionary
            bp_lib: CARLA blueprint library
            enable_visualization: Enable 3D visualization (default: True)
        """
        super().__init__(world, ego_vehicle, sensor_cfg)
        self.lidar = None
        self.vis = None
        
        # Initialize Open3D visualizer if enabled
        if enable_visualization:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=sensor_cfg["sensor_name"],
                width=OPEN3D_VIS_WIDTH,
                height=OPEN3D_VIS_HEIGHT,
                left=OPEN3D_VIS_LEFT,
                top=OPEN3D_VIS_TOP,
            )
            self.vis.get_render_option().background_color = [1.0, 1.0, 1.0]
            self.vis.get_render_option().point_size = OPEN3D_POINT_SIZE
            self.vis.get_render_option().show_coordinate_frame = True
            add_open3d_axis(self.vis)
            self.empty_line_set = o3d.geometry.LineSet()
            self.empty_points = self.empty_line_set.points

        self.processing_func = process_point_cloud
        self.init_sensor(bp_lib)
        
        # Get static bounding boxes (buildings, infrastructure, etc.)
        self.static_bboxes = self.world.get_level_bbs(carla.CityObjectLabel.Car)
        
        # Initialize frame tracking and point cloud data structures
        self.frame = 0
        self.pcd = o3d.geometry.PointCloud()
        self.pcd_save = o3d.t.geometry.PointCloud()
        self.line_sets = []

    def init_sensor(self, bp_lib):
        """
        Initialize LiDAR sensor and attach to ego vehicle.
        
        Args:
            bp_lib: CARLA blueprint library
        """
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        
        # Set LiDAR parameters from configuration
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
        """
        Retrieve LiDAR data and detect 3D bounding boxes.
        
        Args:
            frame_id: Frame number to retrieve data for
            timeout: Timeout for waiting for sensor data
            
        Returns:
            Tuple of (point_cloud, bounding_boxes, point_cloud_tensor)
        """
        # Get processed point cloud data
        points, colors, intensity = super().retrive_data(frame_id, timeout)
        intensity = [[i] for i in intensity]

        # Store in tensor format for efficient I/O
        self.pcd_save.point["positions"] = o3d.core.Tensor(points)
        self.pcd_save.point["intensities"] = o3d.core.Tensor(intensity)

        # Store in geometry format for visualization
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Get list of vehicles for bounding box detection
        if self.vehicles is None:
            self.vehicles = list(self.world.get_actors().filter("vehicle.*"))
        
        # Detect bounding boxes using client-side detection
        bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(
            self.ego_vehicle,
            self.vehicles,
            self.lidar,
            additional_bb=self.static_bboxes,
        )
        bbs = []

        # Process bounding boxes
        for box in bounding_boxes:
            # Flip x-axis for proper coordinate frame alignment
            box[0, :] *= -1

        # Update visualization if enabled
        if self.vis:
            self._update_visualization(bounding_boxes, bbs)

        return self.pcd, bbs, self.pcd_save

    def _update_visualization(self, bounding_boxes, bbs):
        """
        Update Open3D visualization with point cloud and bounding boxes.
        
        Args:
            bounding_boxes: List of detected bounding boxes
            bbs: List to populate with processed bounding boxes
        """
        # Initialize geometries on first frame
        if self.frame == 2:
            self.vis.add_geometry(self.pcd)

            # Add bounding box line sets
            for box in bounding_boxes:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(np.asarray(box.T))
                line_set.lines = o3d.utility.Vector2iVector(BOUNDING_BOX_LINES)
                self.vis.add_geometry(line_set)
                self.line_sets.append(line_set)

        # Update visualization every frame
        if len(self.line_sets) != 0:
            for i, box in enumerate(bounding_boxes):
                # Check if bounding box contains any points (vehicle detection)
                if is_empty(
                    self.pcd,
                    box,
                    threshold=self.sensor_cfg["vehicle_detection_threshold"],
                ):
                    # Clear box visualization if empty
                    self.line_sets[i].points = self.empty_points
                    self.vis.update_geometry(self.line_sets[i])
                    continue
                
                # Update box visualization
                self.line_sets[i].points = o3d.utility.Vector3dVector(
                    np.asarray(box.T)
                )
                self.vis.update_geometry(self.line_sets[i])
                
                # Calculate oriented bounding box and extract yaw angle
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(np.array(box.T))
                )
                r = Rotation.from_matrix(obb.R.copy())
                _, _, yaw = r.as_euler("xyz", degrees=True)
                
                # Create BoundingBox namedtuple
                bb = BoundingBox(
                    obb.center.copy(), obb.extent.copy(), np.deg2rad(yaw)
                )
                bbs.append(bb)

        # Update point cloud and refresh renderer
        self.vis.update_geometry(self.pcd)
        for line_set in self.line_sets:
            self.vis.update_geometry(line_set)

        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(LIDAR_SLEEP_TIME)
        self.frame += 1

    def destroy(self):
        """Destroy LiDAR sensor."""
        self.lidar.destroy()
