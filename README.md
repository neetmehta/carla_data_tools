# CARLA Data Collection Tool - Synthetic Data Generation for Autonomous Driving

Comprehensive tool for generating high-quality synthetic training data for autonomous driving applications using the CARLA simulator. Supports multiple sensor types (RGB cameras, depth cameras, semantic segmentation, LiDAR) with automatic bounding box generation and efficient asynchronous data storage.

<img src="assets/output.gif" alt="CARLA Data Collection Preview" width="1000"/>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
  - [Main Configuration](#main-configuration)
  - [Vehicle & Sensor Configuration](#vehicle--sensor-configuration)
  - [KITTI Format Configuration](#kitti-format-configuration)
- [Sensor Setup](#sensor-setup)
  - [RGB Camera](#rgb-camera)
  - [Depth Camera](#depth-camera)
  - [Semantic Segmentation Camera](#semantic-segmentation-camera)
  - [Instance Segmentation Camera](#instance-segmentation-camera)
  - [LiDAR](#lidar)
- [Data Output Format](#data-output-format)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

Generating training data for autonomous driving systems is challenging due to the cost and time involved in real-world data collection and manual annotation. This project leverages CARLA, an open-source autonomous driving simulator, to generate large-scale synthetic datasets with ground-truth annotations automatically.

**Key Benefits:**
- **Cost-Effective:** Generate unlimited training data without expensive real-world collection
- **Fully Annotated:** Automatic bounding boxes, depth maps, semantic segmentation
- **Controlled Environment:** Adjust weather, traffic, vehicle behavior, and sensor configurations
- **Flexible:** Support for multiple sensor types and configurations
- **Scalable:** Asynchronous I/O for efficient data storage at high capture rates

---

## Features

### Sensor Capabilities
- **RGB Cameras:** Multiple configurable RGB cameras with arbitrary placement and FOV
- **Depth Cameras:** Per-pixel depth estimation for monocular depth prediction tasks
- **Semantic Segmentation:** Per-pixel scene semantic labels (road, vehicle, pedestrian, etc.)
- **Instance Segmentation:** Per-pixel instance-level segmentation for panoptic understanding
- **LiDAR:** 3D point cloud generation with configurable parameters (channels, density, noise)
- **Automatic Annotations:** 2D and 3D bounding boxes with vehicle metadata

### Environment Control
- **Dynamic Weather:** Sunny, cloudy, rainy, wet, foggy conditions (customizable)
- **Traffic Simulation:** Multiple NPC vehicles with realistic behavior
- **Pedestrian Simulation:** Crowd simulation with intelligent walking controllers
- **Time Control:** Full synchronous simulation for deterministic behavior

### Data Management
- **Asynchronous I/O:** Multi-threaded disk writing for efficient storage
- **Frame Skipping:** Configurable capture frequency independent of simulation FPS
- **Velocity-Based Filtering:** Optionally capture only when ego vehicle is moving
- **Organized Output:** Structured directory layout with sensor-specific subdirectories

---

## Architecture

### Project Structure

```
carla_data_tools/
├── main.py                          # Main simulation orchestration script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── cfg/                             # Configuration files
│   ├── config.yaml                  # Global simulation parameters
│   ├── vehicle_cfg.yaml             # Ego vehicle sensor configuration
│   └── kitti_config.yaml            # KITTI format output settings
│
├── src/                             # Source code modules
│   ├── __init__.py
│   ├── world_manager.py             # CARLA world & actor management
│   ├── ego_vehicle.py               # Ego vehicle & sensor orchestration
│   ├── sensor_manager.py            # Sensor data capture & processing
│   ├── pygame_display.py            # Real-time visualization
│   ├── weather.py                   # Weather simulation system
│   ├── bounding_box.py              # Bounding box handling
│   └── utils.py                     # Utility functions & async I/O
│
└── assets/                          # Demo assets (GIFs, images, etc.)
```

### Core Components

**main.py** - Orchestration Engine
- Loads configuration from YAML files
- Initializes CARLA world, actors, and sensors
- Manages main simulation loop with frame timing
- Handles asynchronous disk I/O coordination
- Manages visualization and user input

**world_manager.py** - CARLA World Management
- Connects to CARLA server and loads maps
- Spawns NPC vehicles with traffic manager control
- Spawns pedestrians with AI walking controllers
- Manages weather simulation and updates
- Handles safe actor cleanup on shutdown

**ego_vehicle.py** - Ego Vehicle Management
- Configures and attaches sensors to ego vehicle
- Coordinates sensor data collection
- Manages autopilot and vehicle control

**sensor_manager.py** - Sensor Data Pipeline
- **SensorBase** (Abstract): Common sensor interface
- **CameraSensor**: Handles RGB, depth, and segmentation cameras
  - Processes raw sensor data from CARLA
  - Generates 2D bounding boxes with visualization
  - Exports depth maps and segmentation masks
- **LidarSensor**: Processes 3D point cloud data
  - Converts raw CARLA points to Open3D format
  - Generates 3D bounding boxes
  - Optional 3D visualization

**utils.py** - Asynchronous Data Storage
- **AsyncDiskWriter**: Thread-pool based async file writer
- **capture_data_async()**: Queue sensor data for background writing
- **compute_K()**: Camera intrinsic matrix calculation

---

## Requirements

### System Requirements
- **OS:** Windows or Linux
- **Python:** 3.7.x (required by CARLA 0.9.16)
- **CARLA:** Version 0.9.16
- **GPU:** NVIDIA GPU recommended (CPU mode available but slower)

### Dependencies
```
pygame>=2.0.0           # Visualization
numpy>=1.19.0           # Data processing
PyYAML>=5.3            # Configuration parsing
open3d>=0.12.0         # Point cloud processing
carla==0.9.16          # CARLA simulator (install from build)
```

---

## Installation

### Step 1: Install CARLA Simulator

Download the CARLA 0.9.16 build:
- **Windows:** [CARLA_Latest.zip](https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/Dev/CARLA_Latest.zip)
- **Linux:** [CARLA_Latest.tar.gz](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/Dev/CARLA_Latest.tar.gz)

Extract to your desired location. The repository should be cloned into the `PythonAPI` directory.

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux:
source venv/bin/activate

# Verify Python 3.7.x
python --version
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Start CARLA server (in separate terminal)
cd path/to/CARLA_0.9.16
./CarlaUE4.exe  # Windows
# or
./CarlaUE4.sh   # Linux

# Run a quick test (in another terminal)
cd carla_data_tools
python main.py
```

---

## Quick Start

### Basic Usage

1. **Start CARLA Server** (separate terminal):
   ```bash
   cd CARLA_0.9.16
   ./CarlaUE4.exe  # Windows
   ```

2. **Run Data Collection** (your terminal):
   ```bash
   cd carla_data_tools
   python main.py
   ```

3. **View Results**:
   Data is saved to the directory specified in `config.yaml` under `out_dir`
   ```
   out_dir/
   └── run_Town03_sunny_20260126_143022/
       ├── camera_center_rgb/
       ├── camera_center_depth/
       ├── camera_center_semantic/
       ├── lidar_front/
       └── calibration/
   ```

4. **Visualize Data** (optional):
   Set `sensor_preview: true` in `config.yaml` to see real-time visualization

---

## Configuration Guide

### Main Configuration (`config.yaml`)

```yaml
# Server Connection
server_ip: "localhost"
server_port: 2000
timeout: 120

# Simulation Parameters
fps: 30                              # Simulation frame rate
map: "Town03"                        # CARLA map selection
synchronous: true                   # Enable synchronous mode (recommended)

# Actor Spawning
no_of_vehicles: 50                   # Number of NPC vehicles
no_of_pedestrains: 30                # Number of pedestrians (note: typo in config)
pedestrian_running_percentage: 20    # % of pedestrians that run
pedestrian_crossing_percentage: 30   # % that attempt to cross roads

# Traffic Control
traffic_manager_seed: 0              # Traffic manager random seed
vehicle_damage_disabled: false       # Disable vehicle damage
ignore_lights_percentage: 30         # % vehicles that ignore traffic lights
ignore_signs_percentage: 30          # % vehicles that ignore stop signs

# Data Capture
capture_data: true                   # Enable data recording
capture_frequency: 10                # Frames to capture per second (10 Hz)
dynamic_weather: false               # Enable dynamic weather simulation
weather: "sunny"                     # Static weather preset (ignored if dynamic)

# Output
out_dir: "./output"                  # Output directory for collected data

# Visualization
sensor_preview: true                 # Show real-time sensor visualization
vehicle_preview: false               # Show vehicle positions and routes
lidar_preview: true                  # Show LiDAR point cloud

# Advanced Options
vehicle_detection_threshold: 5       # Min LiDAR points to generate bounding box
use_kitti_format: false              # Export in KITTI format
```

### Vehicle & Sensor Configuration (`vehicle_cfg.yaml`)

Define ego vehicle sensors and their placement:

```yaml
sensors:
  # RGB Camera (front)
  - type: "rgb_camera"
    name: "camera_center_rgb"
    size: [1280, 720]
    fov: 90
    translation: [0.0, 0.0, 1.3]      # [x, y, z] relative to vehicle
    rotation: [0.0, 0.0, 0.0]         # [pitch, yaw, roll]

  # Depth Camera
  - type: "depth_camera"
    name: "camera_center_depth"
    size: [1280, 720]
    fov: 90
    translation: [0.0, 0.0, 1.3]
    rotation: [0.0, 0.0, 0.0]

  # Semantic Segmentation Camera
  - type: "sem_seg_camera"
    name: "camera_center_semantic"
    size: [1280, 720]
    fov: 90
    translation: [0.0, 0.0, 1.3]
    rotation: [0.0, 0.0, 0.0]

  # Instance Segmentation Camera
  - type: "inst_seg_camera"
    name: "camera_center_instance"
    size: [1280, 720]
    fov: 90
    translation: [0.0, 0.0, 1.3]
    rotation: [0.0, 0.0, 0.0]

  # LiDAR
  - type: "lidar"
    name: "lidar_front"
    channels: 64
    points_per_second: 600000
    rotation_frequency: 10
    upper_fov: 10
    lower_fov: -30
    noise_stddev: 0.03
    translation: [0.0, 0.0, 1.3]
    rotation: [0.0, 0.0, 0.0]
```

**Camera Placement Tips:**
- Typical hood position: `translation: [0.0, 0.0, 1.3]`
- Left side camera: `translation: [-0.3, 0.0, 1.3]`
- Right side camera: `translation: [0.3, 0.0, 1.3]`
- Rear camera: `rotation: [0.0, 180.0, 0.0]`

See [CARLA Sensor Documentation](https://carla.readthedocs.io/en/0.9.16/ref_sensors/) for detailed parameter information.

### KITTI Format Configuration (`kitti_config.yaml`)

Configure KITTI dataset output format (if `use_kitti_format: true`):

```yaml
save_calibration: true              # Save camera calibration files
save_3d_boxes: true                 # Save 3D bounding boxes
save_2d_boxes: true                 # Save 2D bounding boxes
class_mapping:                      # Map CARLA classes to KITTI classes
  vehicle: "Car"
  pedestrian: "Pedestrian"
  bicycle: "Cyclist"
```

---

## Sensor Setup

### RGB Camera

**Purpose:** Standard color image for object detection and classification

**Configuration:**
```yaml
- type: "rgb_camera"
  name: "camera_center_rgb"
  size: [1280, 720]        # Resolution [width, height]
  fov: 90                  # Field of view in degrees
  translation: [0.0, 0.0, 1.3]
  rotation: [0.0, 0.0, 0.0]
```

**Output Files:**
- `camera_center_rgb/XXXXXX.png` - RGB image (PNG format)
- `camera_center_rgb/XXXXXX.json` - Camera transformation metadata

**Use Cases:**
- Object detection (YOLO, Faster R-CNN)
- Lane detection
- Traffic sign recognition

---

### Depth Camera

**Purpose:** Per-pixel depth estimation for 3D scene understanding

**Configuration:**
```yaml
- type: "depth_camera"
  name: "camera_center_depth"
  size: [1280, 720]
  fov: 90
  translation: [0.0, 0.0, 1.3]
  rotation: [0.0, 0.0, 0.0]
```

**Output Files:**
- `camera_center_depth/XXXXXX.exr` - Depth map (32-bit float format)
- Depth values represent distance from camera in meters

**Use Cases:**
- Monocular depth estimation
- 3D scene reconstruction
- Obstacle detection

---

### Semantic Segmentation Camera

**Purpose:** Per-pixel semantic class labels (road, sidewalk, vehicle, pedestrian, etc.)

**Configuration:**
```yaml
- type: "sem_seg_camera"
  name: "camera_center_semantic"
  size: [1280, 720]
  fov: 90
  translation: [0.0, 0.0, 1.3]
  rotation: [0.0, 0.0, 0.0]
```

**Output Files:**
- `camera_center_semantic/XXXXXX.png` - Semantic segmentation mask (PNG format)
- Each pixel value represents a semantic class ID

**CARLA Semantic Classes:**
- 0: Road
- 1: Sidewalk
- 2: Building
- 3: Wall
- 4: Fence
- 5: Guard Rail
- 6: Bridge
- 7: Tunnel
- 8: Pole
- 9: Pole Group
- 10: Traffic Light
- 11: Traffic Sign
- 12: Vegetation
- 13: Terrain
- 14: Sky
- 15: Person (Pedestrian)
- 16: Bicycle
- 17: Lanemarking
- 18: Reserved
- 19: Car (Vehicle)
- 20: Truck
- 21: Bus
- 22: Train
- 23: Motorcycle
- 24: Bicycle (vehicle)
- 25: Static
- 26: Dynamic
- 27: Other

**Use Cases:**
- Semantic segmentation networks
- Scene understanding
- Road segmentation

---

### Instance Segmentation Camera

**Purpose:** Per-pixel instance-level segmentation (distinguishes between individual objects)

**Configuration:**
```yaml
- type: "inst_seg_camera"
  name: "camera_center_instance"
  size: [1280, 720]
  fov: 90
  translation: [0.0, 0.0, 1.3]
  rotation: [0.0, 0.0, 0.0]
```

**Output Files:**
- `camera_center_instance/XXXXXX.png` - Instance segmentation mask
- Each pixel value represents a unique instance ID

**Use Cases:**
- Panoptic segmentation
- Instance segmentation networks
- Object boundary detection

---

### LiDAR

**Purpose:** 3D point cloud for autonomous driving perception

**Configuration:**
```yaml
- type: "lidar"
  name: "lidar_front"
  channels: 64                    # Number of laser channels (64, 32, 16)
  points_per_second: 600000       # Total points per revolution
  rotation_frequency: 10          # Rotations per second
  upper_fov: 10                   # Upper field of view (degrees)
  lower_fov: -30                  # Lower field of view (degrees)
  noise_stddev: 0.03              # Gaussian noise std deviation (meters)
  translation: [0.0, 0.0, 1.3]
  rotation: [0.0, 0.0, 0.0]
```

**Output Files:**
- `lidar_front/XXXXXX.pcd` - Point cloud in Open3D PCD format
- `lidar_front/XXXXXX_boxes.json` - 3D bounding boxes (min/max coordinates)

**Point Cloud Format:**
- X, Y, Z coordinates in vehicle reference frame
- Optional: Intensity values for each point
- Format: Binary or ASCII PCD

**Use Cases:**
- 3D object detection (PointNet++, VoxelNet)
- Point cloud segmentation
- Autonomous driving perception

---

## Data Output Format

### Directory Structure

```
output/
└── run_Town03_sunny_20260126_143022/
    ├── camera_center_rgb/              # RGB images
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── camera_center_depth/            # Depth maps
    │   ├── 000000.exr
    │   ├── 000001.exr
    │   └── ...
    ├── camera_center_semantic/         # Semantic segmentation
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── camera_center_instance/         # Instance segmentation
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── lidar_front/                    # 3D point clouds
    │   ├── 000000.pcd
    │   ├── 000000_boxes.json           # 3D bounding boxes
    │   ├── 000001.pcd
    │   └── ...
    ├── calibration/                    # Sensor calibration
    │   ├── camera_center_rgb_K.txt     # Intrinsic matrix
    │   └── transforms.json              # Extrinsic calibration
    └── metadata/                       # Dataset metadata
        ├── config.yaml                 # Simulation parameters
        └── sensor_config.yaml          # Sensor configurations
```

### Bounding Box Format

**2D Bounding Boxes** (from RGB camera):
```json
{
  "frame_id": 0,
  "bounding_boxes": [
    {
      "x1": 100,
      "y1": 200,
      "x2": 250,
      "y2": 450,
      "class": "vehicle",
      "confidence": 1.0,
      "actor_id": 123
    }
  ]
}
```

**3D Bounding Boxes** (from LiDAR):
```json
{
  "frame_id": 0,
  "bounding_boxes_3d": [
    {
      "center": [10.5, 2.3, 0.5],       # [x, y, z] in vehicle frame
      "extent": [4.5, 2.0, 1.7],        # [length, width, height]
      "rotation": 0.45,                 # Yaw angle in radians
      "class": "vehicle",
      "actor_id": 123,
      "velocity": [5.2, 0.1, 0.0]       # [vx, vy, vz]
    }
  ]
}
```

### Calibration Format

**Camera Intrinsic Matrix** (`camera_center_rgb_K.txt`):
```
fx 0  cx
0  fy cy
0  0  1
```

**Extrinsic Calibration** (`transforms.json`):
```json
{
  "cameras": {
    "camera_center_rgb": {
      "translation": [0.0, 0.0, 1.3],
      "rotation": [0.0, 0.0, 0.0]
    }
  },
  "lidar": {
    "lidar_front": {
      "translation": [0.0, 0.0, 1.3],
      "rotation": [0.0, 0.0, 0.0]
    }
  }
}
```

---

## Project Structure

### Main Script (`main.py`)

**Responsibilities:**
- Load configuration files
- Initialize CARLA world and actors
- Setup sensors on ego vehicle
- Run main simulation loop
- Coordinate sensor data collection
- Manage asynchronous disk writing
- Handle visualization and user input

**Key Variables:**
- `delta_tick`: Frame skipping interval (for capture frequency)
- `frame_no`: Current simulation frame
- `save_frame_no`: Number of frames saved
- `writer`: AsyncDiskWriter for background I/O

### World Manager (`world_manager.py`)

**Key Classes:**
- `CarlaWorldManager`: Main world orchestration

**Key Methods:**
- `spawn_actors()`: Spawn NPC vehicles
- `spawn_peds()`: Spawn pedestrians with AI controllers
- `spawn_ego_vehicle()`: Initialize ego vehicle
- `tick()`: Advance simulation by one frame
- `destroy_actors()`: Clean up actors on shutdown

### Ego Vehicle (`ego_vehicle.py`)

**Key Methods:**
- `sensor_setup()`: Configure and attach sensors
- `get_sensor_data()`: Retrieve data from all sensors

### Sensor Manager (`sensor_manager.py`)

**Key Classes:**
- `CameraSensor`: RGB, depth, and segmentation cameras
  - `init_sensor()`: Initialize camera
  - `retrive_data()`: Get camera data and annotations
  - `destroy()`: Clean up camera

- `LidarSensor`: 3D point cloud sensor
  - `init_sensor()`: Initialize LiDAR
  - `retrive_data()`: Get point cloud and bounding boxes
  - `destroy()`: Clean up LiDAR

### Utilities (`utils.py`)

**Key Classes:**
- `AsyncDiskWriter`: Multi-threaded async file writer

**Key Functions:**
- `capture_data_async()`: Queue sensor data for writing
- `compute_K()`: Calculate camera intrinsic matrix

---

## Advanced Usage

### Multi-Camera Setup

Setup multiple cameras at different positions:

```yaml
sensors:
  - type: "rgb_camera"
    name: "camera_center_rgb"
    translation: [0.0, 0.0, 1.3]
    rotation: [0.0, 0.0, 0.0]

  - type: "rgb_camera"
    name: "camera_left_rgb"
    translation: [-0.3, 0.0, 1.3]
    rotation: [0.0, -30.0, 0.0]

  - type: "rgb_camera"
    name: "camera_right_rgb"
    translation: [0.3, 0.0, 1.3]
    rotation: [0.0, 30.0, 0.0]

  - type: "rgb_camera"
    name: "camera_rear_rgb"
    translation: [0.0, 0.0, 1.3]
    rotation: [0.0, 180.0, 0.0]
```

### Dynamic Weather Simulation

Enable random weather changes during collection:

```yaml
dynamic_weather: true              # Enable weather changes
weather: "dynamic"                 # Will cycle through weather presets
```

The system automatically transitions between:
- Sunny
- Rainy
- Cloudy
- Wet
- Foggy

### High-Frequency Data Capture

Collect data at 20 Hz while running simulation at 30 Hz:

```yaml
fps: 30                           # Simulation at 30 Hz
capture_frequency: 20             # Capture at 20 Hz
```

Frame skipping automatically set to: `30 / 20 = 1.5` frames

### Custom Traffic Behavior

Adjust vehicle behavior in `config.yaml`:

```yaml
traffic_manager_seed: 42                # Deterministic traffic
vehicle_damage_disabled: true           # Prevent damage effects
ignore_lights_percentage: 50            # 50% vehicles ignore signals
ignore_signs_percentage: 30             # 30% vehicles ignore signs
```

### Real-Time Visualization

Enable to see data collection in action:

```yaml
sensor_preview: true               # Show camera views
vehicle_preview: true              # Show vehicle positions
lidar_preview: true                # Show point cloud
```

**Controls:**
- `ESC` or `Q`: Quit application
- Close window: Stop collection

---

## Troubleshooting

### Connection Issues

**Problem:** `RuntimeError: failed to connect to CARLA`

**Solution:**
1. Ensure CARLA server is running in a separate terminal
2. Check server IP and port in `config.yaml`
3. Verify firewall isn't blocking port 2000
4. Increase `timeout` value in `config.yaml`

### Out of Memory

**Problem:** Python process uses excessive memory

**Solution:**
1. Reduce `disk_writer_queue_size` in `main.py`
2. Decrease `capture_frequency`
3. Reduce camera resolution in `vehicle_cfg.yaml`
4. Reduce `no_of_vehicles` or `no_of_pedestrains`

### Slow Data Writing

**Problem:** Disk I/O becomes bottleneck

**Solution:**
1. Use faster storage (SSD instead of HDD)
2. Increase `disk_writer_workers` in `main.py`
3. Decrease capture frequency
4. Reduce image resolution

### LiDAR Visualization Not Showing

**Problem:** Point cloud window doesn't appear

**Solution:**
1. Set `lidar_preview: true` in `config.yaml`
2. Ensure Open3D is installed: `pip install open3d`
3. Check that visualization window isn't behind main window

### Map Not Loading

**Problem:** Map specified in `config.yaml` doesn't load

**Solution:**
1. Verify map name is correct (e.g., "Town01", "Town02", "Town03")
2. Check CARLA server has the map available
3. Try loading map from CARLA client directly

### Pedestrians Not Spawning

**Problem:** No pedestrians appear in simulation

**Solution:**
1. Set `no_of_pedestrains` > 0 in `config.yaml`
2. Verify `spawn_peds()` is called in main loop
3. Check CARLA logs for spawn errors
4. Try reducing number of pedestrians if memory-limited

---

## Performance Tips

1. **Synchronous vs Asynchronous:**
   - Use `synchronous: true` for consistent timing
   - Disable for faster simulation (less reliable)

2. **Frame Rate:**
   - Higher FPS = more data but slower processing
   - 20-30 FPS typically recommended
   - Reduce for faster iteration during development

3. **Sensor Resolution:**
   - Smaller resolution = faster processing
   - 1280x720 is good balance
   - Use 640x480 for testing

4. **Disk I/O:**
   - Async writing prevents simulation bottleneck
   - Store data on fast SSD for best performance
   - Compress if storage is limiting factor

5. **Traffic Density:**
   - More vehicles = more computation
   - Start with 30-50 vehicles for testing
   - Scale up for production datasets

---

## References

- [CARLA Documentation](https://carla.readthedocs.io/)
- [CARLA Sensor Reference](https://carla.readthedocs.io/en/0.9.16/ref_sensors/)
- [CARLA Python API](https://carla.readthedocs.io/en/0.9.16/python_api/)
- [KITTI Dataset Format](http://www.cvlibs.net/datasets/kitti/)

---

## Contributing

Found a bug or have a suggestion? Feel free to open an issue or submit a pull request.

## License

This project is provided as-is for research and educational purposes.
