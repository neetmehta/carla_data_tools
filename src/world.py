import glob
import os
import sys
import random
import queue

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


class CarlaWorld:
    """Carla world class"""

    def __init__(self, cfg=None) -> None:
        self.delta_seconds = 1.0 / cfg.get("fps", 20)
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(cfg['map'])
        self.ego_vehicle = None
        self.world_queue = queue.Queue()
        self._settings = None
        self.num_cars = cfg["no_of_vehicles"]

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
        self.spawn_points = self.world.get_map().get_spawn_points()
        random.seed(0)

        if self.num_cars > len(self.spawn_points):
            print("more cars then spawn points")
            self.num_cars = len(self.spawn_points) - 1

        else:
            self.spawn_points = self.spawn_points[: self.num_cars]

        # Select some models from the blueprint library
        models = [
            "dodge",
            "audi",
            "model3",
            "mini",
            "mustang",
            "lincoln",
            "prius",
            "nissan",
            "crown",
            "impala",
        ]
        blueprints = []
        for vehicle in self.world.get_blueprint_library().filter("*vehicle*"):
            if any(model in vehicle.id for model in models):
                blueprints.append(vehicle)

        self.vehicles = []

        # Take a random sample of the spawn points and spawn some vehicles
        for i, spawn_point in enumerate(
            random.sample(self.spawn_points, self.num_cars)
        ):
            temp = self.world.try_spawn_actor(random.choice(blueprints), spawn_point)
            if temp is not None:
                self.vehicles.append(temp)
                temp.set_autopilot(True)

    def tick(self):
        return self.world.tick()

    def destroy_actors(self):
        for actor in self.world.get_actors().filter("*vehicle*"):
            actor.destroy()
        for actor in self.world.get_actors().filter("*sensor*"):
            actor.destroy()
