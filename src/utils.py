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


def capture_data(
    frame_no,
    out_dir,
    sensor_name,
    rgb=None,
    depth=None,
    semantic_mask=None,
    lidar_pc=None,
    bbs=None,
    bb_2d=None,
):
    """captures the data and save it to the given folder

    Args:
        frame_no (int): Frame number in the simulation
    """

    sensor_root = os.path.join(out_dir, sensor_name)

    if rgb is not None:
        os.makedirs(os.path.join(sensor_root, "rgb_images"), exist_ok=True)
        cv2.imwrite(os.path.join(sensor_root, "rgb_images", f"{frame_no}.jpg"), rgb)

    if depth is not None:
        os.makedirs(os.path.join(sensor_root, "depth"), exist_ok=True)
        np.save(os.path.join(sensor_root, "depth", f"{frame_no}.npy"), depth)

    if semantic_mask is not None:
        os.makedirs(os.path.join(sensor_root, "semantic_mask"), exist_ok=True)
        cv2.imwrite(
            os.path.join(sensor_root, "semantic_mask", f"{frame_no}.jpg"), semantic_mask
        )

    if lidar_pc is not None:
        os.makedirs(os.path.join(sensor_root, "lidar"), exist_ok=True)
        o3d.io.write_point_cloud(
            os.path.join(sensor_root, "lidar", f"{frame_no}.pcd"), lidar_pc
        )

    if bbs is not None:
        os.makedirs(os.path.join(sensor_root, "bb_labels"), exist_ok=True)
        with open(os.path.join(sensor_root, "bb_labels", f"{frame_no}.txt"), "w") as f:
            lines = []
            for bb in bbs:
                line = f"Vehicle {0.0} {0} {0.0} {0} {0} {0} {0} {bb.extent[-1]:.{2}f} {bb.extent[-2]:.{2}f} {bb.extent[-3]:.{2}f} {bb.center[0]:.{2}f} {bb.center[1]:.{2}f} {bb.center[2]:.{2}f} {bb.yaw:.{2}f}\n"
                lines.append(line)

            f.writelines(lines)
            f.close()

    if bb_2d is not None:
        os.makedirs(os.path.join(sensor_root, "2d_bb_labels"), exist_ok=True)
        with open(
            os.path.join(sensor_root, "2d_bb_labels", f"{frame_no}.txt"), "w"
        ) as f:
            lines = []
            for bb in bb_2d:
                line = (
                    f"Vehicle {bb[0]:.{2}f} {bb[1]:.{2}f} {bb[2]:.{2}f} {bb[3]:.{2}f}\n"
                )
                lines.append(line)

            f.writelines(lines)
            f.close()


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
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    # array = array[:, :, ::-1]
    return array


def process_depth_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array.astype(np.float32)
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth * 1000


def process_sem_seg_image(image):
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


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c, not_transform=True):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc[0], loc[1], loc[2], 1])
    # transform to camera coordinates
    if not not_transform:
        point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [loc[1], -loc[2], loc[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]
