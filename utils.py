import numpy as np
import glob
import os
import sys
import open3d as o3d
from matplotlib import cm
import cv2

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


def capture_data(frame_no, **kwargs):
    """captures the data and save it to the given folder

    Args:
        frame_no (int): Frame number in the simulation
    """
    out_dir = kwargs.get("out_dir", os.path.join(os.getcwd(), "data"))
    rgb = kwargs.get("rgb", None)
    depth = kwargs.get("depth", None)
    semantic_mask = kwargs.get("semantic", None)
    lidar_pc = kwargs.get("point_cloud", None)
    bb = kwargs.get("bounding_boxes", None)

    if rgb:
        os.makedirs(os.path.join(out_dir, "rgb_images"))
        cv2.imwrite(os.path.join(out_dir, f"{frame_no}.jpg", rgb))

    if depth:
        os.makedirs(os.path.join(out_dir, "depth"))

    if semantic_mask:
        os.makedirs(os.path.join(out_dir, "semantic_mask"))

    if lidar_pc:
        os.makedirs(os.path.join(out_dir, "lidar"))

    if bb:
        os.makedirs(os.path.join(out_dir, "3d_bb"))


def is_empty(pcd, box, threshold=10):
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.min(box, axis=1), max_bound=np.max(box, axis=1)
    )

    filtered_point_cloud = pcd.crop(bounding_box)
    return len(np.array(filtered_point_cloud.points)) < threshold


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    vis.add_geometry(axis)


def retrive_data(sensor_queue, frame_id, timeout):
    while True:
        data = sensor_queue.get(timeout=timeout)
        if data.frame == frame_id:
            return data


def process_rgb_image(image):
    print(f"rgb time {image.timestamp}")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    # array = array[:, :, ::-1]
    return array


def process_depth_image(image):
    print(f"depth time {image.timestamp}")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array.astype(np.float32)
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth


def process_sem_seg_image(image):
    print(f"sem time {image.timestamp}")
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    # array = array[:, :, ::-1]
    return array


def process_point_cloud(point_cloud):

    # Auxilliary code for colormaps and axes
    VIRIDIS = np.array(cm.get_cmap("plasma").colors)
    VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

    COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
    COOL = np.array(cm.get_cmap("winter")(COOL_RANGE))
    COOL = COOL[:, :3]
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype("f4")))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2]),
    ]

    points = data[:, :-1]

    points[:, :1] = -points[:, :1]
    
    return points, int_color
