
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random
import numpy as np

class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """
    @staticmethod
    def get_bounding_boxes(ego_vehicle, vehicles, sensor, additional_bb=None):
        """
        Creates 3D bounding boxes based on carla vehicle list and sensor.
        """

        if additional_bb:
            static_bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box_static(bbox, sensor) for bbox in additional_bb]
            
        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, sensor) for vehicle in vehicles if vehicle.id != ego_vehicle.id]
        
        
        bounding_boxes.extend(static_bounding_boxes)
        
        return bounding_boxes
    
    @staticmethod
    def get_bounding_box(vehicle, sensor):
        """
        Returns 3D bounding box for a vehicle based on sensor view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, sensor)[:3, :]

        return cords_x_y_z
    
    @staticmethod
    def get_bounding_box_static(bbox, sensor):
        """
        Returns 3D bounding box for a vehicle based on sensor view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points_static(bbox)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(bb_cords, sensor)[:3, :]

        return sensor_cord
    
    @staticmethod
    def _create_bb_points_static(bbox):
        """
        Returns 3D bounding box for a carla.BoundingBox object.
        """
        cords = np.zeros((8, 4))
        extent = bbox.get_world_vertices(carla.Transform())
        cords[0, :] = np.array([extent[0].x, extent[0].y, extent[0].z, 1])
        cords[4, :] = np.array([extent[1].x, extent[1].y, extent[1].z, 1])
        cords[3, :] = np.array([extent[2].x, extent[2].y, extent[2].z, 1])
        cords[7, :] = np.array([extent[3].x, extent[3].y, extent[3].z, 1])
        cords[1, :] = np.array([extent[4].x, extent[4].y, extent[4].z, 1])
        cords[5, :] = np.array([extent[5].x, extent[5].y, extent[5].z, 1])
        cords[2, :] = np.array([extent[6].x, extent[6].y, extent[6].z, 1])
        cords[6, :] = np.array([extent[7].x, extent[7].y, extent[7].z, 1])
        return cords.T
    
    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords
    
    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords
    
    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix
