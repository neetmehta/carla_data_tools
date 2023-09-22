# Carla Data Generator - Generating Synthetic Data for Autonomous Driving

##Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Data Generation](#data-generation)
    - [Camera Setup](#camera-setup)
    - [Lidar Setup](#lidar-setup)


## Project Overview

Generating data for training deep learning networks for autonomous driving task is time consuming and expensive process. Annotating that data is yet another challenge. Annotating frames by human is not only time consuming and expensive process but also humans are not a perfect annotators. To overcome this challenges, CARLA has been developed from the ground up to support development, training, and validation of autonomous driving systems. In addition to open-source code and protocols, CARLA has provided open digital assets (urban layouts, buildings, vehicles) that were created for this purpose and can be used freely. The simulation platform supports flexible specification of sensor suites, environmental conditions, full control of all static and dynamic actors, maps generation and much more.

This project is used to generate synthetic data for training different Deep learning models for autonomous driving tasks. This project support cameras and lidars and can generate annotaions for tasks such as 3D lidar object detection, monocular depth estimation, and semantic segmentation.

Feel free to add any comments and suggestions.

## Getting Started

Download and extract latest CARLA build for windows from [here](https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/Dev/CARLA_Latest.zip) and for linux from [here](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/Dev/CARLA_Latest.tar.gz). Extract the zip file and clone this repo inside PythonAPI folder.

You will need Python 3.7.x to run this project. Other versions of python are not supported by carla. It is recommended to create a virtual environment to run this project.

Install all the requirements by running 
```
pip install -r requirements.txt
```

## Data Generation

#### Environment setup
First you have to specify some global parameters like
- fps: FPS of the simulation
- no_of_vehicles: No. of NPC vehicles
- synchronous: set synchronous mode to true. (recommended)
- vehicle_detection_threshold: minimum number of lidar points detected to draw bounding box around it. (If car is in the scene and only 10 points are detected by lidar, the simulator wont generate bounding box if threshold is greater than 10)
- out_dir: Directory in which you want to save the data
- capture_frequency: frequency at which you want capture the frames. (1 means 1 frame per simulation second. 0.5 means 1 frame every 2 second)


To start generating data you first have to setup the sensors on the ego vehicle. You can provide configuration of each sensor to [vehicle_cfg.yaml](cfg/vehicle_cfg.yaml). Currently only lidar and camera are supported. 

#### Camera setup

To setup the camera you have to specify type of camera for eg. depth_camera, sem_seg_camera, rgb_camera. You then have to name the sensor. The name of each sensor should be unique. You also have to specify parameters like size of the image, fov. To get more details about the parameters please refer to the [carla documentation](https://carla.readthedocs.io/en/0.9.14/ref_sensors/#basic-camera-attributes_1). To set the location and rotation of camera sensor with respect to ego vehicle, you have to specify translation and rotaion vector. You will have to do some trial and error to setup the camera to you desired position. To preview your sensor you can change sensor_preview to True in [config.yaml](cfg/config.yaml) and run:

```
python main.py
```

#### Lidar setup

To setup the camera you have to specify type of sensor as lidar. You also have to specify parameters like channels, points_per_second, rotation_frequency, noise_stddev, upper_fov, etc. To get more details about the parameters please refer to the [carla documentation](https://carla.readthedocs.io/en/0.9.14/ref_sensors/#lidar-attributes). To set the location and rotation of lidar sensor with respect to ego vehicle, you have to specify translation and rotaion vector. You will have to do some trial and error to setup the camera to you desired position. To preview your sensor you can change sensor_preview to True in [config.yaml](cfg/config.yaml) and run:

```
python main.py
```