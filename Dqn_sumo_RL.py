
import os
import sys
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# SUMO-RL imports
from sumo_rl import SumoEnvironment
from sumo_rl.environment.traffic_signal import TrafficSignal
from sumo_rl.environment.observations import ObservationFunction
from gymnasium import spaces
import gymnasium as gym



NET_FILE = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\net.net.xml"
ROUTE_FILE = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\rou.route.xml"

# ✅ IMPORTANT: additional file that contains your laneAreaDetector definitions
ADD_FILE = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\network_IA.add.xml"

# Work zone configuration
WORK_ZONE_EDGE = "E#9"
WORK_ZONE_LANE = "E#9_1"
WORK_ZONE_SPEED_LIMIT = 11.11  # 40 km/h

# Detector IDs (must match ids in network_IA.add.xml)
DETECTOR_IDS = ["e_4", "e_5", "e_6", "e_7", "e_8", "e_9", "e_10", "e_11"]

# Traffic Lights
TLS_CONTROLLED = "TL2"  # Only control TL2

# State and Action
STATE_SIZE = 9
ACTION_SIZE = 2

# Work-zone capacity
WORK_ZONE_CAPACITY = (1600 - 200) * 0.90 * 1  # 1260 veh/h

# Normalization constants
MAX_DELAY = 100.0
MAX_STOPS = 50.0
MAX_TTC = 10.0
TTC_THRESHOLD = 1.5

# DQN Hyperparameters
TOTAL_EPISODES = 10
STEPS_PER_EPISODE = 1000
DECISION_INTERVAL = 10

GAMMA = 0.90
LR = 1e-3
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
LEARN_START = 1_000
TARGET_UPDATE_EVERY = 250

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 8_000

# Initialize previous metrics for delta calculation
prev_delay_tl2 = None
prev_stops_tl2 = None


class WorkZoneObservation(ObservationFunction):
    """Custom observation for work zone: 8 detectors + phase."""
    def __init__(self, ts):
        super().__init__(ts)

    def __call__(self):
        import traci

        queues = []
        for det_id in DETECTOR_IDS:
            try:
                q = traci.lanearea.getLastStepVehicleNumber(det_id)
                queues.append(float(q))
            except:
                queues.append(0.0)

        phase = float(self.ts.green_phase)
        state = np.array(queues + [phase], dtype=np.float32)
        return state

    def observation_space(self):
        return spaces.Box(
            low=np.zeros(STATE_SIZE, dtype=np.float32),
            high=np.ones(STATE_SIZE, dtype=np.float32) * 100,
            dtype=np.float32
        )



def work_zone_delta_reward(ts: TrafficSignal):
    global prev_delay_tl2, prev_stops_tl2

    delay = sum(ts.get_accumulated_waiting_time_per_lane())
    stops = ts.get_total_queued()

    ttc_conflicts = 0
    try:
        import traci
        controlled_lanes = traci.trafficlight.getControlledLanes(ts.id)
        controlled_edges = {lane_id.rsplit('_', 1)[0] for lane_id in controlled_lanes}

        for edge_id in controlled_edges:
            try:
                vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                for i in range(len(vehicle_ids) - 1):
                    try:
                        leader_id = vehicle_ids[i]
                        follower_id = vehicle_ids[i + 1]

                        leader_pos = traci.vehicle.getLanePosition(leader_id)
                        follower_pos = traci.vehicle.getLanePosition(follower_id)
                        follower_speed = traci.vehicle.getSpeed(follower_id)
                        leader_speed = traci.vehicle.getSpeed(leader_id)

                        gap = leader_pos - follower_pos
                        relative_speed = follower_speed - leader_speed

                        if relative_speed > 0 and gap > 0:
                            ttc = gap / relative_speed
                            if ttc < TTC_THRESHOLD:
                                ttc_conflicts += 1
                    except:
                        continue
            except:
                continue
    except:
        pass

    if prev_delay_tl2 is None:
        prev_delay_tl2 = delay
        prev_stops_tl2 = stops
        return 0.0

    delta_delay = prev_delay_tl2 - delay
    delta_stops = prev_stops_tl2 - stops

    prev_delay_tl2 = delay
    prev_stops_tl2 = stops

    delay_term = delta_delay / MAX_DELAY
    stops_term = delta_stops / MAX_STOPS
    ttc_term = ttc_conflicts / MAX_TTC

    reward = (0.6 * delay_term + 0.3 * stops_term - 0.5 * ttc_term)
    reward = np.clip(reward, -1.0, 1.0)
    return reward

TrafficSignal.register_reward_fn(work_zone_delta_reward)



class WorkZoneWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.work_zone_setup_done = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self.work_zone_setup_done:
            self.setup_work_zone()
            self.work_zone_setup_done = True
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def setup_work_zone(self):
        try:
            print("\n🚧 Configuring work zone parameters...")
            import traci
            traci.edge.setMaxSpeed(WORK_ZONE_EDGE, WORK_ZONE_SPEED_LIMIT)
            print(f"   ✅ Speed limit on {WORK_ZONE_EDGE} set to {WORK_ZONE_SPEED_LIMIT * 3.6:.0f} km/h")
            print(f"   🎯 Work zone capacity: {WORK_ZONE_CAPACITY:.0f} veh/h\n")
        except Exception as e:
            print(f"   ⚠️ Warning: Could not configure work zone speed: {e}\n")

# -------------------------
# DQN Agent
# -------------------------

def build_q_network():
    model = keras.Sequential([
        layers.Input(shape=(STATE_SIZE,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(ACTION_SIZE, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss="mse")
    return model

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a),
            np.array(r),
            np.array(s2, dtype=np.float32),
            np.array(d),
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self):
        self.q_net = build_q_network()
        self.target_net = build_q_network()
        self.target_net.set_weights(self.q_net.get_weights())
        self.replay = ReplayBuffer(REPLAY_CAPACITY)
        self.train_steps = 0
        self.total_steps = 0

    def epsilon(self):
        if self.total_steps >= EPS_DECAY_STEPS:
            return EPS_END
        return EPS_START + self.total_steps / EPS_DECAY_STEPS * (EPS_END - EPS_START)

    def act(self, state):
        if random.random() < self.epsilon():
            return random.randint(0, ACTION_SIZE - 1)
        state_array = np.array(state, dtype=np.float32).reshape(1, -1)
        q_vals = self.q_net.predict(state_array, verbose=0)[0]
        return int(np.argmax(q_vals))

    def train_one_step(self):
        if len(self.replay) < LEARN_START:
            return

        s, a, r, s2, d = self.replay.sample(BATCH_SIZE)

        q_pred = self.q_net.predict(s, verbose=0)
        q_next = self.target_net.predict(s2, verbose=0)
        max_q_next = np.max(q_next, axis=1)

        for i in range(BATCH_SIZE):
            q_pred[i, a[i]] = r[i] + (1 - d[i]) * GAMMA * max_q_next[i]

        self.q_net.fit(s, q_pred, verbose=0)

        self.train_steps += 1
        if self.train_steps % TARGET_UPDATE_EVERY == 0:
            self.target_net.set_weights(self.q_net.get_weights())



def run_training():

    ADD_FILE = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\network_IA.add.xml"

    additional_sumo_cmd = (
    f'-a "{ADD_FILE}" '
    '--error-log sumo_error.log '
    '--log sumo_run.log '
    '--verbose true'
    )
    
    try:
        base_env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            use_gui=True,
            num_seconds=STEPS_PER_EPISODE,
            delta_time=DECISION_INTERVAL,
            yellow_time=2,
            min_green=5,
            max_green=50,
            single_agent=True,
            reward_fn='work_zone_delta_reward',
            observation_class=WorkZoneObservation,
            sumo_warnings=True,
            additional_sumo_cmd=additional_sumo_cmd
        )

        env = WorkZoneWrapper(base_env)

    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return

    agent = DQNAgent()

    episode_rewards = []
    all_rewards = []
    all_delays = []
    all_stops = []

    for episode in range(TOTAL_EPISODES):
        print(f"\n📍 Episode {episode + 1}/{TOTAL_EPISODES}")
        obs, info = env.reset()

        episode_reward = 0.0
        done = False
        truncated = False
        step = 0

        while not done and not truncated and step < STEPS_PER_EPISODE:
            action = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            agent.replay.push(obs, action, reward, next_obs, done or truncated)
            agent.train_one_step()

            episode_reward += float(reward)
            all_rewards.append(float(reward))

            try:
                all_delays.append(info.get('system_total_waiting_time', 0))
                all_stops.append(info.get('system_total_stopped', 0))
            except:
                pass

            obs = next_obs
            step += 1
            agent.total_steps += 1

            if step % 100 == 0:
                print(f"  Step {step}/{STEPS_PER_EPISODE} | Reward: {episode_reward:.2f} | ε: {agent.epsilon():.3f}")

        episode_rewards.append(episode_reward)
        print(f"✅ Episode {episode + 1} complete | Total Reward: {episode_reward:.2f}")

    env.close()

    model_path = "dqn_sumo_rl_model.h5"
    agent.q_net.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")

    print("\n📊 Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DQN Work Zone Control - SUMO-RL Version', fontsize=14, fontweight='bold')

    axes[0, 0].plot(range(1, TOTAL_EPISODES + 1), episode_rewards, marker='o', linewidth=2)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].grid(True, alpha=0.3)

    if len(all_rewards) > 0:
        axes[0, 1].plot(all_rewards, linewidth=1)
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].set_title("Reward vs Time")
        axes[0, 1].grid(True, alpha=0.3)

    if len(all_delays) > 0:
        axes[1, 0].plot(all_delays, linewidth=1.5)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Total Delay (s)")
        axes[1, 0].set_title("Total Delay Over Time")
        axes[1, 0].grid(True, alpha=0.3)

    if len(all_stops) > 0:
        axes[1, 1].plot(all_stops, linewidth=1.5)
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Number of Stops")
        axes[1, 1].set_title("Total Stops Over Time")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_sumo_rl_results.png', dpi=300, bbox_inches='tight')
    print("✅ Results plot saved to: dqn_sumo_rl_results.png")
    plt.show()

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_training()
