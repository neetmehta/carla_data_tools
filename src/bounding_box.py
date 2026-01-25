import carla
import numpy as np

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

        bounding_boxes = []
        for vehicle in vehicles:
            if vehicle.id != ego_vehicle.id:
                bbox = ClientSideBoundingBoxes.get_bounding_box(vehicle, sensor)
                actor_class = SEMANTIC_MAP[vehicle.semantic_tags[0]][0]
                actor_id = vehicle.id
                bounding_boxes.append(
                    {"bbox": bbox, "actor_class": actor_class, "actor_id": actor_id}
                )

        if additional_bb:
            for static_class_list, class_id in additional_bb:
                for static_object in static_class_list:
                    bbox = ClientSideBoundingBoxes.get_bounding_box_static(
                        static_object, sensor
                    )
                    actor_id = 0
                    actor_class = SEMANTIC_MAP[class_id][0]
                    bounding_boxes.append(
                        {"bbox": bbox, "actor_class": actor_class, "actor_id": actor_id}
                    )

        return bounding_boxes

    @staticmethod
    def get_bounding_box(vehicle, sensor):
        """
        Returns 3D bounding box for a vehicle based on sensor view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(
            bb_cords, vehicle, sensor
        )[:3, :]

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
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(
            vehicle.get_transform()
        )
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
