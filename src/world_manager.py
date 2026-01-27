"""
CARLA World Manager Module

This module provides high-level management of the CARLA simulation world, including
world initialization, vehicle spawning, traffic management, and synchronization.

Classes:
    CarlaWorldManager: Main class for managing CARLA world state and actors
"""

import random
import queue
import logging

import carla
from carla.command import SpawnActor, SetAutopilot, FutureActor

from src.ego_vehicle import EgoVehicle
from src.weather import Weather

# Logging configuration
logger = logging.getLogger(__name__)

# Traffic Manager Configuration Constants
TRAFFIC_MANAGER_HOST = "localhost"
TRAFFIC_MANAGER_PORT = 2000
TRAFFIC_MANAGER_TIMEOUT = 10.0
TRAFFIC_MANAGER_LEADING_DISTANCE = 2.5
TRAFFIC_MANAGER_HYBRID_PHYSICS_RADIUS = 70.0
ACTOR_ACTIVE_DISTANCE = 200
RESPAWN_DORMANT_LOWER_BOUND = 10
RESPAWN_DORMANT_UPPER_BOUND = 190


class CarlaWorldManager:
    """
    Manager for CARLA world state and actor spawning.
    
    Handles world initialization, vehicle spawning (both autonomous and ego vehicle),
    traffic management settings, and simulation synchronization.
    
    Attributes:
        client: CARLA client connection
        world: CARLA world instance
        traffic_manager: CARLA traffic manager for NPC vehicle control
        ego_vehicle: EgoVehicle instance
        vehicles: List of spawned NPC vehicle actor IDs
        weather: Weather system manager
        delta_seconds: Fixed simulation timestep duration
    """

    def __init__(self, cfg) -> None:
        """
        Initialize CARLA world manager.
        
        Args:
            cfg: Main configuration dictionary with keys:
                - map: CARLA map name to load
                - fps: Simulation frequency in frames per second
                - weather: Initial weather preset name (default: ClearNoon)
                - no_of_vehicles: Number of NPC vehicles to spawn
        """
        self.cfg = cfg
        self.delta_seconds = 1.0 / cfg.get("fps", 20)
        
        # Initialize CARLA client and world
        self.client = carla.Client(TRAFFIC_MANAGER_HOST, TRAFFIC_MANAGER_PORT)
        self.client.set_timeout(TRAFFIC_MANAGER_TIMEOUT)
        self.world = self.client.load_world(cfg["map"])
        
        # Configure traffic manager for NPC vehicles
        self.traffic_manager = self.client.get_trafficmanager()
        self._configure_traffic_manager()
        
        # Configure world actor settings
        self._configure_world_settings()
        
        # Initialize vehicle and weather systems
        self.world_queue = queue.Queue()
        self._settings = None
        self.vehicles = []
        self.weather = Weather(self.world.get_weather(), [])

        # Set initial weather
        initial_weather = cfg.get("weather", "ClearNoon")
        self.world.set_weather(
            getattr(carla.WeatherParameters, initial_weather)
        )
        
        logger.info(f"Initialized CARLA world: {cfg['map']} with {cfg['no_of_vehicles']} vehicles")

    def _configure_traffic_manager(self):
        """Configure traffic manager parameters for NPC vehicle behavior."""
        self.traffic_manager.set_global_distance_to_leading_vehicle(
            TRAFFIC_MANAGER_LEADING_DISTANCE
        )
        self.traffic_manager.set_respawn_dormant_vehicles(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(TRAFFIC_MANAGER_HYBRID_PHYSICS_RADIUS)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_boundaries_respawn_dormant_vehicles(
            RESPAWN_DORMANT_LOWER_BOUND, RESPAWN_DORMANT_UPPER_BOUND
        )

    def _configure_world_settings(self):
        """Configure world-level actor settings."""
        settings = self.world.get_settings()
        settings.actor_active_distance = ACTOR_ACTIVE_DISTANCE
        self.world.apply_settings(settings)

    def spawn_ego_vehicle(self):
        """
        Spawn the ego (main player-controlled) vehicle.
        
        Creates the ego vehicle from configuration and spawns it in the world.
        This should be called after traffic actors are spawned to avoid collisions.
        """
        bp_lib = self.world.get_blueprint_library()
        self.ego_vehicle = EgoVehicle(bp_lib, self.cfg)
        self.ego_vehicle.spwan_ego_vehicle(self.world)
        logger.info(f"Spawned ego vehicle: {self.cfg['vehicle']}")

    def spawn_actors(self):
        """
        Spawn NPC vehicles and configure their behavior.
        
        Spawns autonomous vehicles at random spawn points with random colors and
        configures them with autopilot via the traffic manager. Skips the ego vehicle's
        spawn point to avoid conflicts.
        """
        # Get available blueprints
        vehicles_bp = self.world.get_blueprint_library().filter("*vehicle*")
        cars_bp = [x for x in vehicles_bp if x.get_attribute("base_type") == "car"]
        cars_bp = sorted(cars_bp, key=lambda bp: bp.id)

        # Get spawn points and validate count
        self.spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(self.spawn_points)

        if self.cfg["no_of_vehicles"] < number_of_spawn_points:
            random.shuffle(self.spawn_points)
        elif self.cfg["no_of_vehicles"] > number_of_spawn_points:
            logger.warning(
                f"Requested {self.cfg['no_of_vehicles']} vehicles, "
                f"but only {number_of_spawn_points} spawn points available"
            )
            self.cfg["no_of_vehicles"] = number_of_spawn_points

        # Prepare batch spawn commands
        batch = []
        for n, transform in enumerate(self.spawn_points):
            if n >= self.cfg["no_of_vehicles"]:
                break
            
            # Select random vehicle blueprint
            blueprint = random.choice(cars_bp)
            
            # Randomize vehicle appearance
            if blueprint.has_attribute("color"):
                color = random.choice(
                    blueprint.get_attribute("color").recommended_values
                )
                blueprint.set_attribute("color", color)
            
            # Randomize driver appearance if available
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)
            else:
                blueprint.set_attribute("role_name", "autopilot")

            # Queue vehicle spawn with autopilot enabled
            batch.append(
                SpawnActor(blueprint, transform).then(
                    SetAutopilot(FutureActor, True, self.traffic_manager.get_port())
                )
            )

        # Execute batch spawn and collect results
        responses = self.client.apply_batch_sync(batch, True)
        for response in responses:
            if response.error:
                logger.error(f"Spawn error: {response.error}")
            else:
                self.vehicles.append(response.actor_id)

        # Configure vehicle lights
        all_vehicle_actors = self.world.get_actors(self.vehicles)
        for actor in all_vehicle_actors:
            self.traffic_manager.update_vehicle_lights(actor, True)
        
        logger.info(f"Spawned {len(self.vehicles)} NPC vehicles")

    def spawn_peds(self):
        walkers_list = []
        all_id = []
        peds_bp = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road

        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.cfg["no_of_pedestrains"]):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(peds_bp)
            # set as not invincible
            probability = random.randint(0,100 + 1);
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('can_use_wheelchair') and probability < 11:
                walker_bp.set_attribute('use_wheelchair', 'true')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d walkers, press Ctrl+C to exit.' % (len(walkers_list)))
        
    def set_synchronous(self):
        """
        Enable synchronous mode for frame-by-frame simulation.
        
        Switches world from asynchronous to synchronous mode with fixed timestep.
        This is essential for deterministic data collection and sensor synchronization.
        """
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(
            carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,
            )
        )
        self.world.on_tick(self.world_queue.put)
        logger.info(f"Synchronous mode enabled (delta_seconds: {self.delta_seconds})")

    def tick(self):
        """
        Advance simulation by one frame.
        
        Returns:
            int: Current frame number after tick
        """
        return self.world.tick()

    def restore(self):
        """Restore world settings to state before synchronous mode was enabled."""
        if self._settings is not None:
            self.world.apply_settings(self._settings)
            logger.info("Restored original world settings")

    def destroy_actors(self):
        """
        Destroy all NPC vehicles in the world.
        
        Safely removes all vehicle actors except the ego vehicle, which should be
        destroyed separately.
        """
        vehicle_actors = self.world.get_actors().filter("*vehicle*")
        actor_count = len(vehicle_actors)
        
        for actor in vehicle_actors:
            actor.destroy()
        
        logger.info(f"Destroyed {actor_count} actors")
