
[![DOI](https://zenodo.org/badge/161216111.svg)](https://zenodo.org/doi/10.5281/zenodo.10869789)
[![tests](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml/badge.svg)](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml)
[![PyPI version](https://badge.fury.io/py/sumo-rl.svg)](https://badge.fury.io/py/sumo-rl)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/LucasAlegre/sumo-rl/blob/main/LICENSE)

# SUMO-RL: Work Zone Traffic Signal Control with DQN

<!-- start intro -->

SUMO-RL provides a simple interface to instantiate Reinforcement Learning (RL) environments with [SUMO](https://github.com/eclipse/sumo) for Traffic Signal Control.

This fork extends the original SUMO-RL framework with a **multi-model comparison pipeline** that evaluates trained DQN agents against SUMO's default fixed-timing controller in a **work zone / construction scenario**. It measures five key metrics — waiting time, queue length, average speed, throughput, and TTC (Time-To-Collision) safety conflicts — and generates Pareto front visualizations to support multi-objective model selection.

Goals of this repository:
- Provide a simple interface to work with Reinforcement Learning for Traffic Signal Control using SUMO
- Support Multiagent RL
- Compatibility with `gymnasium.Env` and popular RL libraries such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [RLlib](https://docs.ray.io/en/main/rllib.html)
- Easy customisation: state and reward definitions are easily modifiable
- **Evaluate and compare multiple DQN Pareto-optimal agents against the SUMO default controller**
- **Multi-objective analysis: efficiency (waiting, queue, speed, throughput) vs. safety (TTC conflicts)**

The main class is [SumoEnvironment](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py).
If instantiated with parameter `single_agent=True`, it behaves like a regular [Gymnasium Env](https://github.com/Farama-Foundation/Gymnasium).
For multiagent environments, use [env](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) or [parallel_env](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) to instantiate a [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) environment with AEC or Parallel API, respectively.
[TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py) is responsible for retrieving information and actuating on traffic lights using [TraCI](https://sumo.dlr.de/wiki/TraCI) API.

For more details, check the [documentation online](https://lucasalegre.github.io/sumo-rl/).

<!-- end intro -->

---

## Install

<!-- start install -->

### Install SUMO latest version:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

Don't forget to set the `SUMO_HOME` variable (default sumo installation path is `/usr/share/sumo`):
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

> **Performance tip:** For a ~8x speed boost with Libsumo, declare:
> ```bash
> export LIBSUMO_AS_TRACI=1
> ```
> Note: this disables `sumo-gui` and parallel simulations ([more details](https://sumo.dlr.de/docs/Libsumo.html)).

### Install SUMO-RL

Stable release version is available through pip:
```bash
pip install sumo-rl
```

Or install the latest (unreleased) version:
```bash
git clone https://github.com/LucasAlegre/sumo-rl
cd sumo-rl
pip install -e .
```

### Install additional dependencies for the comparison pipeline:

```bash
pip install stable-baselines3 matplotlib numpy
```

<!-- end install -->

---

## MDP — Observations, Actions and Rewards

### Observation

<!-- start observation -->

The default observation for each traffic signal agent is a vector:
```python
obs = [phase_one_hot, min_green, lane_1_density, ..., lane_n_density, lane_1_queue, ..., lane_n_queue]
```
- `phase_one_hot` — one-hot encoded vector of the current active green phase
- `min_green` — binary flag indicating whether the minimum green time has elapsed
- `lane_i_density` — number of vehicles in incoming lane `i` divided by lane capacity
- `lane_i_queue` — number of queued vehicles (speed < 0.1 m/s) in lane `i` divided by lane capacity

You can define your own observation by implementing a class that inherits from [ObservationFunction](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/observations.py) and passing it to the environment constructor.

<!-- end observation -->

### Rewards

<!-- start reward -->

The reward signal combines multiple competing objectives relevant to work-zone traffic control through weighted scalarization:

<p align="center">
<img src="reward.png" align="center" width="60%"/>
</p>

That is, the reward reflects metrics for accumulated waiting time, queued vehicle count, average speed, traffic pressure, safety conflicts (time-to-collision), and work-zone queue spillback, respectively.

You can choose a different reward function (see the ones implemented in [TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py)) via the `reward_fn` parameter, or define your own:

```python
def my_reward_fn(traffic_signal):
    return traffic_signal.get_average_speed()

env = SumoEnvironment(..., reward_fn=my_reward_fn)
```

<!-- end reward -->

---

## Multi-Model Comparison: DQN Pareto Agents vs SUMO Default

The script [`compare_models.py`](experiments/compare_models.py) evaluates a set of pre-trained DQN models against SUMO's default fixed-timing controller across multiple episodes and seeds.

### Evaluated Models

| Model | Description |
|---|---|
| `SUMO_Default` | Baseline: SUMO's built-in fixed signal timing (no RL) |
| `DQN_w04` | DQN Pareto agent — weight configuration 04 |
| `DQN_w11` | DQN Pareto agent — weight configuration 11 |
| `DQN_w17` | DQN Pareto agent — weight configuration 17 |
| `DQN_w18` | DQN Pareto agent — weight configuration 18 |
| `DQN_w19` | DQN Pareto agent — weight configuration 19 |

### Measured Metrics

| Metric | Description |
|---|---|
| **Waiting Time (s)** | Average cumulative waiting time across all controlled lanes |
| **Queue Length** | Average number of halted vehicles per lane (speed < 0.1 m/s) |
| **Average Speed (m/s)** | Mean vehicle speed across all controlled lanes |
| **Throughput** | Number of vehicles that successfully completed their trip |
| **TTC Conflicts** | Number of vehicles with a Time-To-Collision below 1.5 seconds (safety proxy) |

### Running the Comparison

1. Place your trained model files under `DQN_OUTPUTS/`:
```
DQN_OUTPUTS/
├── dqn_workzone_w04.zip
├── dqn_workzone_w11.zip
├── dqn_workzone_w17.zip
├── dqn_workzone_w18.zip
└── dqn_workzone_w19.zip
```

2. Update the network and route file paths in the script:
```python
NET_FILE   = "path/to/your/net.net.xml"
ROUTE_FILE = "path/to/your/rou.route.xml"
```

3. Run the comparison:
```bash
python experiments/compare_models.py
```

### Configuration

```python
SIMULATION_SECONDS = 200   # Total simulation time per episode
DELTA_TIME         = 5     # Agent decision interval (seconds)
EPISODE_SEEDS      = [11, 22, 33]  # Random seeds for reproducibility
CONTROLLED_TLS_ID  = "TL1"         # Traffic light ID in your network
```

### Output

The script prints a summary table and saves three plots:

| File | Description |
|---|---|
| `comparison_bars.png` | Bar charts for each metric across all models |
| `pareto_2d.png` | 2D Pareto front: Waiting Time vs TTC Conflicts |
| `pareto_3d.png` | 3D Pareto front: Waiting Time vs Queue Length vs TTC Conflicts |

Example console output:
```
====================================================================================================
TRAFFIC SIGNAL CONTROL COMPARISON
====================================================================================================
Model               Waiting (s)    Queue       Speed (m/s)    Throughput     TTC
----------------------------------------------------------------------------------------------------
SUMO_Default        42.31          8.14        3.21           112.00         5.00
DQN_w04             29.18          5.67        4.05           131.00         2.33
DQN_w11             27.40          5.12        4.22           138.00         1.67
...

🏆 BEST PERFORMERS:
  Lowest Waiting:       DQN_w11 (27.40s)
  Lowest TTC Conflicts: DQN_w19 (1.33)
  Highest Speed:        DQN_w17 (4.38 m/s)
  Highest Throughput:   DQN_w11 (138.00)
```

---

## API's (Gymnasium and PettingZoo)

### Gymnasium Single-Agent API

<!-- start gymnasium -->

If your network has only **one traffic light**, instantiate a standard Gymnasium env:
```python
import gymnasium as gym
import sumo_rl

env = gym.make('sumo-rl-v0',
               net_file='path_to_your_network.net.xml',
               route_file='path_to_your_routefile.rou.xml',
               out_csv_name='path_to_output.csv',
               use_gui=True,
               num_seconds=100000)

obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
```

<!-- end gymnasium -->

### Multi-model Pareto comparison (work zone):
```bash
python experiments/compare_models.py
```


<p align="center">
<img src="outputs/result.png" align="center" width="50%"/>
</p>

---



