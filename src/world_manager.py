import glob
import os
import sys
import random
import queue
import logging


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
from carla.command import SpawnActor, SetAutopilot, FutureActor, DestroyActor

from src.ego_vehicle import EgoVehicle
from src.weather import Weather


class CarlaWorldManager:
    """Carla world class"""

    def __init__(self, cfg, vehicle_cfg) -> None:
        self.cfg = cfg
        self.delta_seconds = 1.0 / cfg.get("fps", 20)
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(100.0)
        self.world = self.client.load_world(cfg["map"])
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_respawn_dormant_vehicles(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(70.0)
        self.traffic_manager.set_synchronous_mode(True)
        settings = self.world.get_settings()
        settings.actor_active_distance = 200
        self.traffic_manager.set_boundaries_respawn_dormant_vehicles (10,190)
        self.world.apply_settings(settings)

        self.ego_vehicle_cfg = vehicle_cfg
        self.world_queue = queue.Queue()
        self._settings = None
        self.num_cars = cfg["no_of_vehicles"]
        self.vehicles = []
        self.weather = Weather(self.world.get_weather(), [])

        self.world.set_weather(getattr(carla.WeatherParameters, cfg.get("weather", "ClearNoon")))

    def spawn_ego_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        self.ego_vehicle = EgoVehicle(bp_lib, self.ego_vehicle_cfg)
        self.ego_vehicle.spwan_ego_vehicle(self.world)
        
    def set_synchronous(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(
            carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,
            )
        )
        self.world.on_tick(self.world_queue.put)

    def restore(self):
        self.world.apply_settings(self._settings)

    def spawn_actors(self):
        """spawns npc into the environment"""
        vehicles_bp = self.world.get_blueprint_library().filter("*vehicle*")
        walkers_bp = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        cars_bp = [x for x in vehicles_bp if x.get_attribute('base_type') == 'car']
        cars_bp = sorted(cars_bp, key=lambda bp: bp.id)
        
        self.spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(self.spawn_points)
        
        if self.cfg["no_of_vehicles"] < number_of_spawn_points:
            random.shuffle(self.spawn_points)
        elif self.cfg["no_of_vehicles"] > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.cfg["no_of_vehicles"], number_of_spawn_points)
            self.cfg["no_of_vehicles"] = number_of_spawn_points

        batch = []
        for n, transform in enumerate(self.spawn_points):
            if n >= self.cfg["no_of_vehicles"]:
                break
            blueprint = random.choice(cars_bp)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles.append(response.actor_id)

        # Set automatic vehicle lights update if specified

        all_vehicle_actors = self.world.get_actors(self.vehicles)
        for actor in all_vehicle_actors:
            self.traffic_manager.update_vehicle_lights(actor, True)
            
    def tick(self):
        return self.world.tick()

    def destroy_actors(self):
        for actor in self.world.get_actors().filter("*vehicle*"):
            actor.destroy()
