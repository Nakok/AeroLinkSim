import os
import numpy as np
import torch
import random
import gymnasium as gym
from gymnasium import spaces
import airsim
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import logging
import msgpack
import tornado
import gc
import zmq
import atexit
import json
import time
import datetime
from collections import deque
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


@dataclass(frozen=True)
class State:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float>
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.fl>
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float>
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=>
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=n>

    def to_array(self) -> np.ndarray:
        try:
            return np.concatenate([
                self.position,
                self.orientation,
                self.velocity,
                self.angular_velocity,
                self.target_position
            ], dtype=np.float32)
            except Exception as e:
            logging.error(f"Error serializing state to array: {e}")
            return np.zeros(15, dtype=np.float32)

    @staticmethod
    def from_array(array: np.ndarray) -> 'State':
        try:
            if len(array) != 15:
                raise ValueError(f"Expected array of length 15, got {len(array)}")
            position = array[:3]
            orientation = array[3:6]
            velocity = array[6:9]
            angular_velocity = array[9:12]
            target_position = array[12:]
            return State(position, orientation, velocity, angular_velocity, target_>
        except Exception as e:
            logging.error(f"Error deserializing array to state: {e}")
            return State()

    @staticmethod
    def serialize(state: 'State') -> bytes:
        try:
            state_dict = {
                'position': state.position.tolist(),
                'orientation': state.orientation.tolist(),
                'velocity': state.velocity.tolist(),
                'angular_velocity': state.angular_velocity.tolist(),
                'target_position': state.target_position.tolist()
            }
            return msgpack.packb(state_dict, use_bin_type=True)
        except Exception as e:
            logging.error(f"Error serializing state: {e}")
            return b''

    @staticmethod
    def deserialize(data: bytes) -> 'State':
        try:
            if not data or not isinstance(data, bytes):
                raise ValueError("Provided data is not valid bytes")

            state_dict = msgpack.unpackb(data, raw=False)
            # Validate the unpacked dictionary
            required_keys = ['position', 'orientation', 'velocity', 'angular_velocity', 'target_position']
            if not all(key in state_dict for key in required_keys):
                raise ValueError("Missing keys in unpacked data")

            # Ensure correct data types
            for key in required_keys:
                if not isinstance(state_dict[key], list) or len(state_dict[key]) != 3:
                    raise ValueError(f"Invalid data for key '{key}'")

            return State(
                position=np.array(state_dict['position'], dtype=np.float32),
                orientation=np.array(state_dict['orientation'], dtype=np.float32),
                velocity=np.array(state_dict['velocity'], dtype=np.float32),
                angular_velocity=np.array(state_dict['angular_velocity'], dtype=np.float32),
                target_position=np.array(state_dict['target_position'], dtype=np.float32)
            )
        except msgpack.exceptions.UnpackValueError as e:
            logging.error(f"Error unpacking data: {e}")
            return State()
        except ValueError as ve:
            logging.error(f"Value error during deserialization: {ve}")
            return State()
        except Exception as e:
            logging.error(f"Error deserializing state: {e}")
            return State()



class AirSimEnv(gym.Env):
    def __init__(self, N=20, pub_address="tcp://*:5556", sub_address="tcp://localhost:5556",
                 action_pub_address="tcp://*:5558", action_sub_address="tcp://localhost:5558"):
        super(AirSimEnv, self).__init__()

        # Connect to AirSim
        self._connect_to_airsim()

        # Gym spaces
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32)
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(N, 15),  # 12 kinematics + 3 target positions
            dtype=np.float32
        )

        # ZMQ setup
        self.pub_address = pub_address
        self.sub_address = sub_address
        self.context = zmq.Context()

        # Publisher and subscriber sockets
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(self.pub_address)

        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(self.sub_address)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

        # Action sockets
        self.action_pub_address = action_pub_address
        self.action_sub_address = action_sub_address

        self.action_publisher = self.context.socket(zmq.PUB)
        self.action_publisher.bind(self.action_pub_address)

        self.action_subscriber = self.context.socket(zmq.SUB)
        self.action_subscriber.connect(self.action_sub_address)
        self.action_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        # General environment parameters
        self.N = N
        self.movement_interval = 0.01
        self.last_print_time = time.time()
        self.print_interval = 1
        self.OUT_OF_BOUNDS_LIMIT = np.array([200, 200, 50])
        self.noise_stddev = 0.1
        self.max_steps = 1000

        # Curriculum learning variables
        self.episode_metrics = []
        self.file_name = "episode_metrics.json"
        self.episode_count = 0
        self.curriculum_stage = 1
        self.stage_length = 5
        self.difficulty_increment = 10
        self.min_target_distance = 20.0
        self.max_target_distance = 200.0

        # Tracking variables
        self.steps = 0
        self.current_stage = 0
        self.current_stage_target_count = 0
        self.targets_reached = 0
        self.success_count = 0
        self.mean_reward = 0
        self.total_steps = 0
        self.episodes = 0
        self.episode_length = 0
        self.episode_count = 1

        # Buffers for performance tracking
        self.observation_window = deque(maxlen=N)
        self.performance_buffer = deque(maxlen=100)
        self.distance_buffer = deque(maxlen=100)

        # Target-related variables
        self.start_position = airsim.Vector3r(0, 0, -35)
        self.current_target = None
        self.target_position = None
        self.current_target_index = 0
        self.target_positions = []
        self.target_success_count = [0] * self.stage_length
        # Success rate thresholds
        self.success_rate_threshold = 0.7
        self.regression_threshold = 0.4

        self.simulated_metrics = {
            "rtt_action": None,
            "processing_latency": None,
            "zmq_delay": None,
            "latency": None,
            "jitter": None,
            "data_transferred": 0,
            "packet_loss": False,
            "packet_loss_obs": False,
            "message_duplication": False,
            "message_corruption": False,
            # Initialize lists to track metrics over time
            "latencies": [],
            "jitters": [],
            "packet_losses": 0,
            "packet_losses_obs": 0,
            "message_duplications": 0,
            "message_corruptions": 0,
            "rtt_obs": [],
            "latency_obs": [],
            "packet_size_obs": [],
            "propagation_delays": [],
            "transmission_delays": [],
            "environmental_noises": []

        }
        self.training_speeds = []
        self.throughputs = []
        self.obs_size = 15
        self.last_valid_observation = np.zeros(self.obs_size, dtype=np.float32)

        # State initialization
        self.state = State(
            position=np.zeros(3, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            target_position=np.zeros(3, dtype=np.float32)
        )
        self.prev_velocity = None
        self.prev_orientation = None
        # Rendering
        self.render_enabled = False

        # Initialize stage and environment
        self.reset()

        # Register cleanup on exit
        atexit.register(self.close)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir='logs')


    def _connect_to_airsim(self):
        try:
            print("Connecting...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        except Exception as e:
            logging.error(f"Error connecting to AirSim: {e}")
            raise

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def _send_action_to_drone(self, pitch, roll, yaw, throttle):
        """Publish action to ZeroMQ."""
        try:
            action_message = np.array([pitch, roll, yaw, throttle], dtype=np.float32)
            serialized_message = action_message.tobytes()
            message_size = len(serialized_message)

            # Timestamp for send time
            send_time = time.time()

            # Send action message
            self.action_publisher.send(serialized_message)

            # Debug information
            #print(f"[DEBUG] Sent action: pitch={pitch}, roll={roll}, yaw={yaw}, throttle={throttle}")
            #print(f"[DEBUG] Serialized message size: {message_size} bytes")
            #print(f"[DEBUG] Action sent at: {send_time}")

            return send_time, message_size
        except Exception as e:
            print(f"[ERROR] Failed to send action to drone: {e}")
            return None, 0


    def _receive_action_ack(self, send_time):
        """Receive acknowledgment and calculate RTT and End-to-End Latency."""
        try:
            #print("[DEBUG] Waiting for acknowledgment...")

            # Poll for a response with a timeout
            poll_start = time.time()  # Time before polling
            if self.action_subscriber.poll(timeout=100):  # 100 ms timeout
                ack_message = self.action_subscriber.recv()
                recv_time = time.time()  # Time of acknowledgment receipt

                # Detailed delay calculations
                poll_delay = poll_start - send_time  # Delay due to polling start
                rtt_action = recv_time - send_time  # Round Trip Time (RTT)
                processing_latency = recv_time - poll_start  # Time between polling start and receipt

                # Debug information
                #print(f"[DEBUG] Acknowledgment received.")
                #print(f"[DEBUG] Poll delay: {poll_delay:.5f} seconds")
                #print(f"[DEBUG] RTT: {rtt_action:.5f} seconds")
                #print(f"[DEBUG] Processing latency: {processing_latency:.5f} seconds")

                return rtt_action, processing_latency
            else:
                print("[WARNING] No acknowledgment received within the timeout.")
                return None, None
        except Exception as e:
            print(f"[ERROR] Failed to receive acknowledgment: {e}")
            return None, None


    def reset(self, seed=None, options=None):
        # Ensure `options` is a dictionary
        options = options or {}

        # Reset environment variables
        self.steps = 0
        self.targets_reached = 0
        self.current_stage_target_count = 0

        # Preserve the target if it hasn't been reached yet
        if options.get("force_reset_target", False) or self.target_position is None:
            self.current_stage = 0  # Reset to the first stage
            self.target_position = self._generate_target(self.current_stage)
        #logging.info(f"Environment reset. Target position preserved: {self.target_position}")

        # Clear buffers
        self.performance_buffer.clear()
        self.observation_window.clear()

        # Seed RNG if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset drone
        try:
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
            self.client.hoverAsync().join()
            logging.info("Drone successfully reset and hovering.")
        except Exception as e:
            logging.error(f"Error during reset operations: {e}. Retrying...")
            return self.reset(seed, options)

        # Set starting position
        start_position = airsim.Vector3r(0, 0, -35)
        try:
            self.client.simSetVehiclePose(airsim.Pose(start_position, airsim.Quaternionr(0, 0, 0, 1)), True)
            #logging.info(f"Drone starting position set to: {start_position}")
        except Exception as e:
            logging.error(f"Error setting vehicle pose: {e}. Using default position.")
            self.client.simSetVehiclePose(airsim.Pose(start_position, airsim.Quaternionr(0, 0, 0, 1)), True)    
        # Initialize observations
        try:
            state_array = self._get_obs()
            self.state = State.from_array(state_array)
            self.prev_distance = np.linalg.norm(self.state.position - self.target_position)
            #logging.info(f"Initial observation acquired. Distance to target: {self.prev_distance:.2f}")
        except Exception as e:
            logging.error(f"Error getting initial observation: {e}")
            self.state = State(
                position=np.zeros(3, dtype=np.float32),
                velocity=np.zeros(3, dtype=np.float32),
                orientation=np.zeros(3, dtype=np.float32),
                angular_velocity=np.zeros(3, dtype=np.float32),
                target_position=self.target_position
            )
            self.prev_distance = np.inf

        # Fill the observation window
        for _ in range(self.N):
            self.observation_window.append(self.state.to_array())
        #logging.info(f"Observation window initialized with {self.N} observations.")

        return np.array(self.observation_window), {}



    def _simulate_latency(self, bandwidth=10e6, packet_size=512):
        """
        Simulates latency and returns individual components: propagation, transmission, and environm>
        """
        # Dynamic bandwidth based on congestion
        network_congestion_factor = np.random.normal(1.0, 0.1)  # Mean=1.0, std dev=0.1
        adjusted_bandwidth = bandwidth * max(0.5, min(network_congestion_factor, 1.5))  # Cap extrem>

        # Distance and propagation delay
        distance = np.linalg.norm(self.state.position - self.target_position) if hasattr(self.state,>
        propagation_delay = distance / 3e8  # Speed of light in m/s

        # Transmission delay
        transmission_delay = packet_size * 8 / adjusted_bandwidth

        # Environmental noise
        environmental_noise = random.uniform(0.0, 0.002)

        # Total latency
        latency = propagation_delay + transmission_delay + environmental_noise

        # Log individual components
        self.simulated_metrics.setdefault("propagation_delays", []).append(propagation_delay)
        self.simulated_metrics.setdefault("transmission_delays", []).append(transmission_delay)
        self.simulated_metrics.setdefault("environmental_noises", []).append(environmental_noise)
        self.simulated_metrics.setdefault("latencies", []).append(latency)

        return latency

    def _simulate_jitter(self, base_latency=0.003, jitter_factor=0.001):
        """
        Simulates jitter dynamically and logs the individual jitter value.
        """
        # Drone velocity and angular velocity
        velocity = np.linalg.norm(self.state.velocity) if hasattr(self.state, 'velocity') else 0
        angular_velocity = np.linalg.norm(self.state.angular_velocity) if hasattr(self.state, 'angular_velocity') else 0

        # Jitter based on movement and time-varying factors
        movement_factor = (velocity + angular_velocity) / 20
        time_factor = np.sin(time.time() * 2 * np.pi / 60)  # 60-second cycle
        jitter = jitter_factor * (1 + movement_factor + time_factor)

        # Apply jitter to latency
        jittered_latency = max(0, random.uniform(base_latency - jitter, base_latency + jitter))
        self.simulated_metrics.setdefault("jitters", []).append(jittered_latency)

        return jittered_latency


    def _simulate_packet_loss(self, base_loss_rate=0.005):
        """
        Simulates packet loss dynamically based on signal strength.
        """
        distance = np.linalg.norm(self.state.position - self.target_position) if hasattr(self.state, 'position') else 200
        signal_strength = max(0, 1 - (distance / 200))  # Signal degrades over distance
        loss_rate = base_loss_rate * (1 - signal_strength)

        packet_lost = random.random() <= loss_rate
        if packet_lost:
            self.simulated_metrics["packet_losses"] = self.simulated_metrics.get("packet_losses", 0) + 1
            logging.warning("Simulated packet loss occurred!")
        return not packet_lost



    def _simulate_message_duplication(self, base_duplication_rate=0.002):
        """
        Simulates message duplication dynamically based on network traffic.
        """
        network_traffic_factor = random.uniform(0.8, 1.5)  # Higher traffic increases duplication chance
        duplication_rate = base_duplication_rate * network_traffic_factor

        message_duplicated = random.random() <= duplication_rate
        if message_duplicated:
            self.simulated_metrics["message_duplications"] = self.simulated_metrics.get("message_duplications", 0) + 1
            logging.warning("Simulated message duplication occurred!")
        return message_duplicated

    def _simulate_message_corruption(self, message, corruption_rate=0.0005):
        """
        Simulates message corruption by randomly altering the message with a given probability.

        Parameters:
        - message (bytes): The original message to be sent.
        - corruption_rate (float): Probability of message corruption (default: 0.05%).

        Returns:
        - bytes: The potentially corrupted message.
        """
        # Ensure simulated_metrics dictionary exists
        if not hasattr(self, "simulated_metrics"):
            self.simulated_metrics = {}

        if random.random() <= corruption_rate:
            self.simulated_metrics["message_corruptions"] = self.simulated_metrics.get("message_corruptions", 0) + 1
            corrupted_message = bytearray(message)
            corrupted_message[0] ^= 1  # Flip a bit in the first byte to simulate corruption
            logging.warning("Simulated message corruption occurred!")
            return bytes(corrupted_message)
        return message   


    def step(self, action):
        start_time = time.time()  # Start timing the step
        if self.steps == 0:
            self.rewards_per_step = []
        self.steps += 1  # Increment step count

        step_data_transferred = 0
        rtt_action = None
        processing_latency = None
        zmq_delay = None
        send_time = None  # Initialize send_time to None

        # Interpret action
        pitch, roll, yaw, throttle = self._interpret_action(action)

        # Communication simulation
        try:
            # Simulate network latency and jitter
            latency = self._simulate_latency()
            jitter = self._simulate_jitter(latency)
            time.sleep(latency + jitter)  # Simulate network delay

            # Append simulated latencies and jitters to the lists in simulated_metrics
            self.simulated_metrics["latencies"].append(latency)  # Append latency to the list
            self.simulated_metrics["jitters"].append(jitter)    # Append jitter to the list

            # Simulate packet loss
            if not self._simulate_packet_loss():
                self.simulated_metrics["packet_loss"] = True
                self.simulated_metrics["packet_losses"] += 1
                raise Exception("Simulated packet loss")

            # Send action and measure ZeroMQ delay
            send_start = time.time()
            send_time, message_size = self._send_action_to_drone(pitch, roll, yaw, throttle)
            zmq_delay = time.time() - send_start  # Measure ZMQ delay
            self.simulated_metrics["zmq_delay"] = zmq_delay
            self.simulated_metrics["data_transferred"] = message_size
            step_data_transferred += message_size  # Accumulate data transferred

            # Simulate message duplication
            if self._simulate_message_duplication():
                self.simulated_metrics["message_duplication"] = True
                self.simulated_metrics["message_duplications"] += 1
                # Send action again to simulate duplication
                self._send_action_to_drone(pitch, roll, yaw, throttle)

            # Handle response and calculate RTT and processing latency
            if send_time:
                rtt_start = time.time()
                rtt_action, processing_latency = self._receive_action_ack(send_time)

                # Ensure RTT is not None and update the metric
                if rtt_action is not None:
                    self.simulated_metrics["rtt_action"] = rtt_action
                    self.simulated_metrics["latencies"].append(rtt_action)  # Append RTT to latencies list
                else:
                    logging.warning("RTT calculation failed, setting RTT to None")
                    self.simulated_metrics["rtt_action"] = None

                self.simulated_metrics["processing_latency"] = processing_latency
                self.simulated_metrics["latencies"].append(processing_latency)  # Append processing latency

                # Simulate message corruption
                corrupted_response = self._simulate_message_corruption(b"Response")
                if corrupted_response != b"Response":
                    self.simulated_metrics["message_corruption"] = True
                    self.simulated_metrics["message_corruptions"] += 1
                    raise Exception("Simulated corrupted response")

        except Exception as e:
            logging.error(f"Communication error: {e}")

        # Calculate throughput (data transferred / elapsed time)
        elapsed_time = time.time() - start_time
        step_throughput = step_data_transferred / elapsed_time if elapsed_time > 0 else 0

        # Enforce movement interval (simulate real-world delays)
        time.sleep(self.movement_interval)

        # Get observation from the environment
        try:
            state_array = self._get_obs()
            self.state = State.from_array(state_array)
            position = self.state.position
        except Exception as e:
            logging.error(f"[ERROR] Failed to get observation: {e}")
            position = np.zeros(3, dtype=np.float32)

        # Calculate distance to target
        distance_to_target = np.linalg.norm(position - self.target_position)

        # Check if the target is reached
        if distance_to_target < 8.0:
            self.targets_reached += 1

            if self.current_target_index < len(self.target_positions):
                self.target_success_count[self.current_target_index] += 1
            else:
                logging.warning(f"current_target_index {self.current_target_index} out of bounds.")
                self.current_target_index = 0  # Ensure the index is within bounds

            logging.info(f"Target reached at position {position}. Distance: {distance_to_target:.2f}")

            self._log_target_reached()  # Log the target reached event

            # Handle stage progression
            if self.current_stage_target_count + 1 >= self.stage_length:
                # If all targets in the stage have been reached, advance to the next stage
                self.current_stage += 1
                self.current_stage_target_count = 0  # Reset target count for the new stage
                logging.info(f"Stage {self.current_stage - 1} completed. Advancing to Stage {self.current_stage}.")

                # Reset the target index and generate the first target for the new stage
                self.current_target_index = 0
            else:
                # Increment the counters if there are more targets in the current stage
                self.current_stage_target_count += 1
                self.current_target_index += 1

            # Generate a new target for the current stage
            self.target_position = self._generate_target(self.current_stage)
            logging.info(f"New target generated: {self.target_position}")

        # Ensure minimum altitude
        self._ensure_minimum_altitude(position, pitch, roll, yaw, throttle)

        # Calculate reward
        reward, target_reached = self._calculate_reward()
        self.rewards_per_step.append(reward)

        # Check if the episode is done
        done = self.steps >= self.max_steps

        # Step timing and throughput metrics
        elapsed_time = time.time() - start_time
        training_speed = 1 / elapsed_time if elapsed_time > 0 else 0  # Steps per second
        step_throughput = step_data_transferred / elapsed_time if elapsed_time > 0 else 0  # Bytes per second

        # Accumulate step-level metrics for averaging
        self.training_speeds.append(training_speed)
        self.throughputs.append(step_throughput)

        # Store all info in the metrics dictionary
        info = {
            "targets_reached": self.targets_reached,
            "distance_to_target": distance_to_target,
            "current_stage": self.current_stage + 1,
            "rtt_action": self.simulated_metrics["rtt_action"],
            "processing_latency": self.simulated_metrics["processing_latency"],
            "zmq_delay": self.simulated_metrics["zmq_delay"],
            "end_to_end_latency": elapsed_time if send_time else None,
            "step_training_speed": training_speed,
            "step_throughput": step_throughput,
        }

        # Periodic logging for step metrics
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            logging.info("--------------- STEP METRICS ---------------")
            logging.info(f"STEP: {self.steps} | Current Stage: {self.current_stage + 1}")
            logging.info(f"Current Position: {self.state.position}")
            logging.info(f"Target Position: {self.target_position}")
            logging.info(f"Distance to Target: {distance_to_target:.2f}")
            logging.info(f"Reward: {reward:.2f}")
            logging.info(f"RTT: {self.simulated_metrics['rtt_action'] if self.simulated_metrics['rtt_action'] is not None else 'N/A'}")
            logging.info(f"Processing Latency: {self.simulated_metrics['processing_latency'] if self.simulated_metrics['processing_latency'] is not None else 'N/A'}")
            logging.info(f"ZeroMQ Delay: {self.simulated_metrics['zmq_delay'] if self.simulated_metrics['zmq_delay'] is not None else 'N/A'}")
            logging.info(f"Step Training Speed: {training_speed:.2f} steps/sec")
            logging.info(f"Step Throughput: {step_throughput:.2f} bytes/sec")

            self.last_print_time = current_time

        # Log episode metrics at the end of an episode
        if done:
            self.mean_reward = np.mean(self.rewards_per_step) if self.rewards_per_step else 0
            logging.info(f"Mean reward for Episode {self.episodes + 1}: {self.mean_reward:.2f}")
            self._log_episode_metrics()
            self.episodes += 1  # Increment the episode counter
            logging.info(f"--- Episode {self.episodes} Completed ---")

            # Reset per-step metric accumulators for the next episode
            self.training_speeds.clear()
            self.throughputs.clear()
            self.rewards_per_step = []

        # Append the current state to the observation window
        self.observation_window.append(self.state.to_array())
        stacked_obs = np.array(self.observation_window)

        return stacked_obs, reward, done, False, info



    def _generate_target(self, stage):
        # Base starting position
        start_x, start_y, start_z = self.start_position.x_val, self.start_position.y_val, self.start_position.z_val

        if stage < 5:
            # Easy stages: targets within 10m, fixed height
            radius = 10 * stage  # Stage 1: 10m, Stage 4: 40m
            z_variation = 0  # Fixed height, no vertical variation
        elif stage < 10:
            # Medium stages: targets within 20m * stage, slight vertical variation
            radius = 20 * (stage - 4)  # Stage 5: 20m, Stage 9: 100m
            z_variation = 5  # Vertical variation of ±5m
        else:
            # Hard stages: larger radius and more vertical variation
            radius = 50 * (stage - 9)  # Stage 10: 50m, Stage 15: 250m
            z_variation = 10  # Vertical variation of ±10m

        # Generate random target position within the radius
        angle = np.random.uniform(0, 2 * np.pi)  # Random direction in radians
        distance = np.sqrt(np.random.uniform(0, 1)) * radius  # Adjusted for uniform distribution

        # Compute x, y offsets based on the angle and distance
        x_offset = distance * np.cos(angle)
        y_offset = distance * np.sin(angle)

        # Compute target z-coordinate with optional variation
        target_x = start_x + x_offset
        target_y = start_y + y_offset
        target_z = start_z + np.random.uniform(-z_variation, z_variation)

        return np.array([target_x, target_y, target_z], dtype=np.float32)

    def _get_stage_length(self, stage):
        """Defines the number of targets in each stage."""
        return 3 + stage

    def _ensure_minimum_altitude(self, position, pitch, roll, yaw, throttle):
        altitude = -position[2]  # Assuming negative z-axis is altitude
        min_altitude = 35.0

        if altitude < min_altitude:
            throttle += 0.5 * (min_altitude - altitude)  # Increase throttle proportionally
            try:
                #print(f"[DEBUG] Adjusting altitude: Current altitude={altitude:.2f}, Throttle={throttle:.2f}")
                self.client.moveByRollPitchYawrateThrottleAsync(
                    pitch, roll, yaw, throttle, duration=self.movement_interval
                ).join()
            except Exception as e:
                print(f"[ERROR] Altitude adjustment failed: {e}")
                # Emergency fallback
                self._retry_altitude_adjustment()


    def _retry_altitude_adjustment(self):
        try:
            #print("[CRITICAL] Retrying altitude adjustment with default throttle.")
            self.client.moveByRollPitchYawrateThrottleAsync(
                0, 0, 0, 0.5, duration=self.movement_interval
            ).join()
        except Exception as retry_e:
            #print(f"[CRITICAL] Failed to adjust altitude during retry: {retry_e}")
            # Consider resetting the environment if this fails
            return self.reset(), -100, True, False, {}



    def _log_target_reached(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure that the position of the state is correctly calculated and up-to-date
        if self.state is not None and self.state.position is not None and self.target_position is not None:
            distance_to_target = np.linalg.norm(self.state.position - self.target_position)
        else:
            distance_to_target = "N/A"

        # Prepare log data
        log_data = {
            "Episode": self.episode_count,
            "Target Index": self.current_target_index + 1,
            "Total Steps": self.total_steps,
            "Distance to Target": distance_to_target
        }

        log_file_path = 'target_log.txt'

        log_entry = (
            f"\n{'-'*40}\n"
            f" Timestamp        : {timestamp}\n"
            f" Episode          : {log_data['Episode']}\n"
            f" Target Index     : {log_data['Target Index']}\n"
            f" Total Steps      : {log_data['Total Steps']}\n"
            f" Distance to Target: {log_data['Distance to Target'] if isinstance(distance_to_target, str) else log_data['Distance to Target']:.2f} units\n"
            f"{'-'*40}\n"
        )

        # Write the log entry to the log file
        with open(log_file_path, 'a') as file:
            file.write(log_entry)

        print(f"Logged target reached:\n{log_entry}")


    def _log_episode_metrics(self):
        """
        Logs detailed metrics at the end of an episode, including averages for latency,
        jitter, throughput, training speed, RTT, data transferred, and packet-related metrics.
        All time-based metrics are reported in seconds. Metrics are also written to a JSON file.
        """
        if not self.training_speeds or not self.throughputs:
            logging.warning("No metrics to log for this episode. Training speeds or throughputs are empty.")
            return

        # Calculate averages for training speed, throughput, RTT, latency, jitter, and ZeroMQ delay
        avg_training_speed = sum(self.training_speeds) / len(self.training_speeds)
        avg_throughput = sum(self.throughputs) / len(self.throughputs)

        # Convert RTT to seconds and calculate average (assuming RTT values are already in seconds)
        rtt_values = self.simulated_metrics.get("rtt_action", [])
        rtt_values = rtt_values if isinstance(rtt_values, list) else [rtt_values] if rtt_values else []
        avg_rtt = (sum(rtt_values) / len(rtt_values)) if rtt_values else None

        # Convert latencies to seconds and calculate average (assuming latencies are in seconds)
        latency_values = self.simulated_metrics.get("latencies", [])
        avg_latency = (sum(latency_values) / len(latency_values)) if latency_values else None

        # Convert jitters to seconds and calculate average (assuming jitters are in seconds)
        jitter_values = self.simulated_metrics.get("jitters", [])
        avg_jitter = (sum(jitter_values) / len(jitter_values)) if jitter_values else None

        # Convert ZeroMQ delay to seconds and calculate average (assuming ZeroMQ delays are in seconds)
        zmq_delay_values = self.simulated_metrics.get("zmq_delay", [])
        zmq_delay_values = zmq_delay_values if isinstance(zmq_delay_values, list) else [zmq_delay_values] if zmq_delay_values else []
        avg_zmq_delay = (sum(zmq_delay_values) / len(zmq_delay_values)) if zmq_delay_values else None

        #OBS##
        obs_rtt_values = self.simulated_metrics.get("rtt_obs", [])
        avg_rtt_obs = (sum(obs_rtt_values) / len(obs_rtt_values)) if obs_rtt_values else None

        obs_latency_values = self.simulated_metrics.get("latency_obs", [])
        avg_latency_obs = (sum(obs_latency_values) / len(obs_latency_values)) if obs_latency_values else None

        packet_sizes = self.simulated_metrics.get("packet_size_obs", [])
        mean_packet_size = (sum(packet_sizes) / len(packet_sizes)) if packet_sizes else 0

        #INDIVIDUAL DELAYS#
        # Extract individual delay components
        propagation_values = self.simulated_metrics.get("propagation_delays", [])
        avg_propagation_delay = (sum(propagation_values) / len(propagation_values)) if propagation_values else None

        transmission_values = self.simulated_metrics.get("transmission_delays", [])
        avg_transmission_delay = (sum(transmission_values) / len(transmission_values)) if transmission_values else None

        environmental_values = self.simulated_metrics.get("environmental_noises", [])
        avg_environmental_noise = (sum(environmental_values) / len(environmental_values)) if environmental_values else None



        # Ensure data transferred is handled as a list
        data_transferred_values = self.simulated_metrics.get("data_transferred", [])
        data_transferred_values = data_transferred_values if isinstance(data_transferred_values, list) else [data_transferred_values] if data_transferred_values else []
        total_data_transferred = sum(data_transferred_values) if data_transferred_values else 0

        # Count total packet-related issues
        total_packet_loss = self.simulated_metrics.get("packet_losses", 0)
        total_packet_loss_obs = self.simulated_metrics.get("packet_losses_obs", 0)
        total_duplications = self.simulated_metrics.get("message_duplications", 0)
        total_corruptions = self.simulated_metrics.get("message_corruptions", 0)

        # Create a dictionary of metrics to log
        episode_metrics = {
            "episode": self.episodes,
            "mean_reward": self.mean_reward,
            "steps": self.steps,
            "targets_reached": self.targets_reached,
            "average_training_speed": avg_training_speed,
            "average_throughput": avg_throughput,
            "average_rtt_seconds": avg_rtt,
            "average_observation_rtt_seconds": avg_rtt_obs,
            "mean_packet_size": mean_packet_size,
            "average_latency_seconds": avg_latency,
            "average_observation_latency_seconds": avg_latency_obs,
            "average_jitter_seconds": avg_jitter,
            "average_zmq_delay_seconds": avg_zmq_delay,
            "average_propagation_delay_seconds": avg_propagation_delay,
            "average_transmission_delay_seconds": avg_transmission_delay,
            "average_environmental_noise_seconds": avg_environmental_noise,
            "total_data_transferred": total_data_transferred,
            "packet_metrics": {
                "total_packet_losses": total_packet_loss,
                "total_packet_losses_obs": total_packet_loss_obs,
                "total_message_duplications": total_duplications,
                "total_message_corruptions": total_corruptions
            },
        }

        # Log detailed episode metrics
        logging.info("\n========== EPISODE METRICS ==========")
        logging.info(f"Episode {self.episodes}")
        logging.info(f"Mean Reward: {self.mean_reward:.2f}")
        logging.info(f"Steps: {self.steps}")
        logging.info(f"Targets Reached: {self.targets_reached}")
        logging.info("-------------------------------------")
        logging.info(f"Average Training Speed: {avg_training_speed:.2f} steps/sec")
        logging.info(f"Average Throughput: {avg_throughput:.2f} bytes/sec")

        if avg_rtt is not None:
            logging.info(f"Average RTT: {avg_rtt:.6f} seconds")
        else:
            logging.warning("RTT data unavailable.")

        if avg_latency is not None:
            logging.info(f"Average Latency: {avg_latency:.6f} seconds")
        else:
            logging.warning("Latency data unavailable.")

        if avg_jitter is not None:
            logging.info(f"Average Jitter: {avg_jitter:.6f} seconds")
        else:
            logging.warning("Jitter data unavailable.")

        if avg_zmq_delay is not None:
            logging.info(f"Average ZeroMQ Delay: {avg_zmq_delay:.6f} seconds")
        else:
            logging.warning("ZeroMQ delay data unavailable.")

        # Include in logs
        if avg_rtt_obs is not None:
            logging.info(f"Average Observation RTT: {avg_rtt_obs:.6f} seconds")
        else:
            logging.warning("Observation RTT data unavailable.")

        if avg_latency_obs is not None:
            logging.info(f"Average Observation Latency: {avg_latency_obs:.6f} seconds")
        else:
            logging.warning("Observation latency data unavailable.")

        # Log individual components
        if avg_propagation_delay is not None:
            logging.info(f"Average Propagation Delay: {avg_propagation_delay:.6f} seconds")
        else:
            logging.warning("Propagation delay data unavailable.")

        if avg_transmission_delay is not None:
            logging.info(f"Average Transmission Delay: {avg_transmission_delay:.6f} seconds")
        else:
            logging.warning("Transmission delay data unavailable.")

        if avg_environmental_noise is not None:
            logging.info(f"Average Environmental Noise: {avg_environmental_noise:.6f} seconds")
        else:
            logging.warning("Environmental noise data unavailable.")

        logging.info(f"Obs Packet Size: {mean_packet_size:.2f} bytes")

        logging.info(f"Total Data Transferred: {total_data_transferred} bytes")

        logging.info("Packet Metrics:")
        logging.info(f"  Total Packet Losses: {total_packet_loss}")
        logging.info(f"  Total Packet Losses OBS {total_packet_loss_obs}")
        logging.info(f"  Total Message Duplications: {total_duplications}")
        logging.info(f"  Total Message Corruptions: {total_corruptions}")

        logging.info(f"Current Stage: {self.current_stage + 1}")
        logging.info("=====================================")

        # Clear metrics accumulators for the next episode
        self.simulated_metrics["latencies"] = []
        self.simulated_metrics["jitters"] = []
        self.simulated_metrics["zmq_delay"] = []
        self.simulated_metrics["rtt_action"] = []
        self.simulated_metrics["data_transferred"] = []
        self.training_speeds.clear()
        self.throughputs.clear()
        self.simulated_metrics["packet_losses"] = 0
        self.simulated_metrics["packet_losses_obs"] = 0
        self.simulated_metrics["message_duplications"] = 0
        self.simulated_metrics["message_corruptions"] = 0

        self.simulated_metrics["rtt_obs"] = []
        self.simulated_metrics["latency_obs"] = []
        self.simulated_metrics["propagation_delays"] = []
        self.simulated_metrics["transmission_delays"] = []
        self.simulated_metrics["environmental_noises"] = []

        logging.info(f"Episode {self.episodes} metrics logged and accumulators reset.")

        # Ensure the directory exists and write the metrics to a JSON file
        file_path = './episode_metrics.json'
        if not os.path.exists(file_path):
            # Create a new file if it doesn't exist
            with open(file_path, 'w') as f:
                json.dump([episode_metrics], f, indent=4)
        else:
            # If the file exists, append the new episode metrics
            with open(file_path, 'r+') as f:
                data = json.load(f)
                data.append(episode_metrics)
                f.seek(0)
                json.dump(data, f, indent=4)



    def _get_obs(self):
        try:
            # Retry logic for fetching data from AirSim
            pose, multirotor_state = None, None
            for attempt in range(3):
                pose = self.client.simGetVehiclePose()
                multirotor_state = self.client.getMultirotorState().kinematics_estimated
                if pose is not None and multirotor_state is not None:
                    break
                logging.warning(f"Attempt {attempt + 1}: Failed to fetch data from AirSim.")
                time.sleep(1)

            if pose is None or multirotor_state is None:
                logging.error("Failed to fetch data from AirSim after 3 attempts.")
                return self.last_valid_observation if hasattr(self, 'last_valid_observation') else None

            # Extract data
            position = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
            orientation = [pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val]
            velocity = [multirotor_state.linear_velocity.x_val, multirotor_state.linear_velocity.y_val, multirotor_state.linear_velocity.z_val]
            angular_velocity = [multirotor_state.angular_velocity.x_val, multirotor_state.angular_velocity.y_val, multirotor_state.angular_velocity.z_val]

            # Create observation dictionary
            obs_dict = {
                "position": position,
                "orientation": orientation,
                "velocity": velocity,
                "angular_velocity": angular_velocity,
                "target_position": self.target_position.tolist()
            }

            # Serialize observation and calculate packet size
            try:
                serialized_obs = json.dumps(obs_dict)
                packet_size = len(serialized_obs.encode('utf-8'))  # Packet size in bytes
                self.simulated_metrics.setdefault("packet_size_obs", []).append(packet_size)

                # Simulate latency and jitter
                latency_obs = self._simulate_latency(bandwidth=10e6, packet_size=packet_size)
                jitter_obs = self._simulate_jitter(base_latency=latency_obs)

                # Send observation
                start_time = time.time()
                self.publisher.send_string(serialized_obs)
                end_time = time.time()

                latency_obs += end_time - start_time  # Account for serialization and send time
                self.simulated_metrics.setdefault("latency_obs", []).append(latency_obs)
            except json.JSONEncodeError as e:
                logging.error(f"Serialization error: {e}")
                return self.last_valid_observation if hasattr(self, 'last_valid_observation') else None

            # Simulate packet loss when receiving observation##############################
            if not self._simulate_packet_loss():
                logging.warning("Simulated packet loss: Falling back to last valid observation.")
                self.simulated_metrics["packet_loss_obs"] = True
                self.simulated_metrics["packet_losses_obs"] += 1
                return self.last_valid_observation
            ######################################

            # Receive and process observation
            try:
                start_time = time.time()
                received_obs = self.subscriber.recv_string(flags=zmq.NOBLOCK)
                end_time = time.time()

                deserialized_obs = json.loads(received_obs)
                logging.debug(f"Deserialized observation: {deserialized_obs}")

                observation = np.concatenate([
                    np.array(deserialized_obs["position"], dtype=np.float32),
                    np.array(deserialized_obs["orientation"], dtype=np.float32),
                    np.array(deserialized_obs["velocity"], dtype=np.float32),
                    np.array(deserialized_obs["angular_velocity"], dtype=np.float32),
                    np.array(deserialized_obs["target_position"], dtype=np.float32)
                ])

                # Validate observation shape
                expected_size = len(position) + len(orientation) + len(velocity) + len(angular_velocity) + len(self.target_position)
                if observation.shape[0] != expected_size:
                    logging.error(f"Unexpected observation shape: {observation.shape}")
                    return self.last_valid_observation if hasattr(self, 'last_valid_observation') else None

                # Calculate round-trip time (RTT)
                rtt_obs = (end_time - start_time) * 2 + jitter_obs  # RTT includes jitter
                self.simulated_metrics.setdefault("rtt_obs", []).append(rtt_obs)

                # Log observation details
                logging.debug(f"Final observation: {observation}")
                current_position = np.array(deserialized_obs["position"], dtype=np.float32)
                distance_to_target = np.linalg.norm(current_position - self.target_position)

                # Update the last valid observation
                self.last_valid_observation = observation
                return observation

            except zmq.Again:
                logging.warning("No observation received from ZeroMQ subscriber. Using last valid observation.")
                return self.last_valid_observation if hasattr(self, 'last_valid_observation') else None
            except json.JSONDecodeError as e:
                logging.error(f"Deserialization error: {e}. Using last valid observation.")
                return self.last_valid_observation if hasattr(self, 'last_valid_observation') else None

        except Exception as e:
            logging.error(f"Unexpected error in _get_obs: {type(e).__name__}: {e}")
            return self.last_valid_observation if hasattr(self, 'last_valid_observation') else None


    def _add_noise(self, action):
        try:
            noise = np.random.normal(0, self.noise_stddev, size=action.shape)
            noisy_action = np.clip(action + noise, self.action_space.low, self.action_space.high)
            return noisy_action
        except Exception as e:
            logging.error(f"Error adding noise to action: {e}")
            return np.clip(action, self.action_space.low, self.action_space.high)  # Fallback

    def _interpret_action(self, action):
        try:
            # Define scaling factors and limits
            PITCH_SCALE = np.pi / 12
            ROLL_SCALE = np.pi / 12
            YAW_SCALE = np.pi / 12
            THROTTLE_SCALE = 0.5

            MAX_PITCH = np.pi / 6
            MAX_ROLL = np.pi / 6
            MAX_YAW = np.pi / 6

            # Scale actions
            pitch = max(-MAX_PITCH, min(MAX_PITCH, action[0] * PITCH_SCALE))
            roll = max(-MAX_ROLL, min(MAX_ROLL, action[1] * ROLL_SCALE))
            yaw = max(-MAX_YAW, min(MAX_YAW, action[2] * YAW_SCALE))
            throttle = min(1.0, max(0.0, 0.5 * (action[3] + 1.0)))

            # Apply smaller random perturbations for smoother flight
            perturbation_scale = 0.005  # Smaller perturbation for smoother response
            pitch += np.random.uniform(-perturbation_scale, perturbation_scale)
            roll += np.random.uniform(-perturbation_scale, perturbation_scale)
            yaw += np.random.uniform(-perturbation_scale, perturbation_scale)

            # Store previous values for smoothing (exponential moving average)
            alpha = 0.9
            self.prev_pitch = alpha * self.prev_pitch + (1 - alpha) * pitch if hasattr(self, 'prev_pitch') else pitch
            self.prev_roll = alpha * self.prev_roll + (1 - alpha) * roll if hasattr(self, 'prev_roll') else roll
            self.prev_yaw = alpha * self.prev_yaw + (1 - alpha) * yaw if hasattr(self, 'prev_yaw') else yaw

            return self.prev_pitch, self.prev_roll, self.prev_yaw, throttle
        except Exception as e:
            print(f"Error interpreting action: {e}")
            return 0.0, 0.0, 0.0, 0.5

    def calculate_heading(self, orientation):
        if isinstance(orientation, np.ndarray):
            if orientation.shape == (3,):
                norm = np.linalg.norm(orientation)
                if norm == 0:
                    print("Warning: Orientation vector has zero magnitude.")
                    return np.array([1, 0, 0])  # Default heading
                return orientation / norm

            elif orientation.shape == (3, 3):
                print(f"Rotation matrix provided. Forward direction: {orientation[:, 0]}")
                return orientation[:, 0]

            else:
                raise ValueError(f"Invalid shape for orientation: {orientation.shape}, expected (3,) or (3, 3).")

        elif isinstance(orientation, (list, tuple)):
            if len(orientation) == 4:
                print(f"Quaternion provided: {orientation}")
                return self._quat_to_heading(orientation)
            else:
                raise ValueError("Quaternion should have 4 components (w, x, y, z).")

        else:
            raise TypeError(f"Unsupported type for orientation: {type(orientation)}")

    def _quat_to_heading(self, quat):
        rotation = R.from_quat(quat)
        heading = rotation.apply([1, 0, 0])
        print(f"Heading from quaternion: {heading}")
        return heading


    def _calculate_reward(self) -> tuple[float, bool]:
        # Constants for reward components
        DISTANCE_REWARD_SCALING = 30
        PROXIMITY_REWARD_SCALING = 150
        SPEED_PENALTY_SCALING = 0.02
        ANGULAR_VELOCITY_PENALTY_SCALING = 0.1
        TIME_PENALTY_SCALING = 0.005  # Adjusted for smoother time penalties
        ORIENTATION_REWARD_SCALING = 15
        SMOOTHNESS_PENALTY_SCALING = 0.5
        TARGET_REWARD = 100
        MIN_REWARD, MAX_REWARD = -100, 150

        # Constants for proximity thresholds
        CLOSE_PROXIMITY_THRESHOLD = 12
        VERY_CLOSE_PROXIMITY_THRESHOLD = 8

        # Extract state information
        state_position = np.array(self.state.position, dtype=np.float32)
        current_target = np.array(self.target_position, dtype=np.float32)

        # Validate state and target positions
        if state_position.size == 0 or current_target.size == 0:
            raise ValueError("Invalid state or target position.")

        # Calculate the distance to the target
        distance_to_target = np.linalg.norm(state_position - current_target)

        # Get velocity and angular velocity of the agent
        velocity = getattr(self.state, 'velocity', np.zeros(3))
        angular_velocity = getattr(self.state, 'angular_velocity', np.zeros(3))

        # Collision check
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            print(f"[DEBUG] Collision detected. Ending episode with MIN_REWARD.")
            return MIN_REWARD, True  # Collision ends the episode

        # Calculate distance reward
        if self.prev_distance is not None and self.prev_distance != float('inf'):
            distance_reward = (self.prev_distance - distance_to_target) * DISTANCE_REWARD_SCALING
        else:
            distance_reward = 0

        # Calculate proximity reward
        proximity_reward = PROXIMITY_REWARD_SCALING * max(0, 5 - distance_to_target)
        if distance_to_target < CLOSE_PROXIMITY_THRESHOLD:
            proximity_reward += 80 * (CLOSE_PROXIMITY_THRESHOLD - distance_to_target)
        if distance_to_target < VERY_CLOSE_PROXIMITY_THRESHOLD:
            proximity_reward += 40 * (VERY_CLOSE_PROXIMITY_THRESHOLD - distance_to_target)

        # Speed penalty for excessive speed
        speed_penalty = SPEED_PENALTY_SCALING * np.linalg.norm(velocity)

        # Angular velocity penalty for excessive rotational speed
        angular_velocity_penalty = ANGULAR_VELOCITY_PENALTY_SCALING * (np.linalg.norm(angular_velocity) ** 1.5)

        # Orientation reward
        orientation = np.array(self.state.orientation, dtype=np.float32)
        if np.linalg.norm(orientation) == 0:
            #print("[DEBUG] Zero orientation vector. Using default orientation.")
            orientation = np.array([1, 0, 0], dtype=np.float32)
        else:
            orientation /= np.linalg.norm(orientation)  # Normalize orientation

        heading = self.calculate_heading(orientation)
        heading = heading / np.linalg.norm(heading) if np.linalg.norm(heading) != 0 else np.array([1, 0, 0])
        target_direction = current_target - state_position
        target_direction = target_direction / np.linalg.norm(target_direction) if np.linalg.norm(target_direction) != 0 else np.array([1, 0, 0])
        dot_product = np.clip(np.dot(heading, target_direction), -1.0, 1.0)
        orientation_error = np.arccos(dot_product)
        orientation_reward = ORIENTATION_REWARD_SCALING * (1 - (orientation_error / np.pi))

        # Smoothness penalty for abrupt changes in velocity, angular velocity, or orientation
        smoothness_penalty = 0
        if self.prev_velocity is not None and self.prev_angular_velocity is not None:
            velocity_change = np.linalg.norm(velocity - self.prev_velocity)
            angular_velocity_change = np.linalg.norm(angular_velocity - self.prev_angular_velocity)
            orientation_change_penalty = 0.3 * np.abs(orientation_error)
            smoothness_penalty = SMOOTHNESS_PENALTY_SCALING * (velocity_change + angular_velocity_change + orientation_change_penalty)

        # Time penalty for episodes that take longer
        time_penalty = TIME_PENALTY_SCALING * (self.steps ** 0.5)  # Square root scaling

        # Calculate the total reward
        reward = (distance_reward + proximity_reward + orientation_reward
                  - speed_penalty - angular_velocity_penalty - smoothness_penalty - time_penalty)

        # Check if the target has been reached
        done = distance_to_target < VERY_CLOSE_PROXIMITY_THRESHOLD
        if done:
            reward += TARGET_REWARD  # Bonus for reaching the target

        # Ensure the reward is within the valid range
        reward = np.clip(reward, MIN_REWARD, MAX_REWARD)

        # Update previous state for next iteration
        self.prev_distance = distance_to_target
        self.prev_velocity = velocity
        self.prev_angular_velocity = angular_velocity

        # Debug logs for key metrics
        #logging.info("----------- [DEBUG LOGS] -----------")
        #logging.info(f"[DEBUG] Distance Reward: {distance_reward:.2f}")
        #logging.info(f"[DEBUG] Proximity Reward: {proximity_reward:.2f}")
        #logging.info(f"[DEBUG] Orientation Reward: {orientation_reward:.2f}")
        #logging.info(f"[DEBUG] Speed Penalty: {speed_penalty:.2f}")
        #logging.info(f"[DEBUG] Angular Velocity Penalty: {angular_velocity_penalty:.2f}")
        #logging.info(f"[DEBUG] Smoothness Penalty: {smoothness_penalty:.2f}")
        #logging.info(f"[DEBUG] Time Penalty: {time_penalty:.2f}")
        #logging.info(f"[DEBUG] Total Reward: {reward:.2f}")


        return reward, done

    def _quaternion_to_forward_vector(self, quaternion):
        # Extract components from quaternion
        x, y, z, w = quaternion

        # Calculate forward direction based on quaternion formula
        forward_x = 1 - 2 * (y**2 + z**2)
        forward_y = 2 * (x * y + z * w)
        forward_z = 2 * (x * z - y * w)

        # Combine into a vector and normalize
        forward_vector = np.array([forward_x, forward_y, forward_z])

        # Normalize the forward vector to ensure it's a unit vector
        return forward_vector / np.linalg.norm(forward_vector)



    def render(self, mode='human'):
        """
        Render the environment.
        """
        if not self.render_enabled:
            logging.info("Render mode is disabled.")
            return
        try:
            # Code to render environment if needed
            pass
        except Exception as e:
            logging.error(f"Error in render method: {e}")

    def close(self):
        self.publisher.close()
        self.subscriber.close()
        self.context.term()        
