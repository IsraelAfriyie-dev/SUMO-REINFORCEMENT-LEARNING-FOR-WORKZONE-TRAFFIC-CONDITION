

[![DOI](https://zenodo.org/badge/161216111.svg)](https://zenodo.org/doi/10.5281/zenodo.10869789)
[![tests](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml/badge.svg)](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml)
[![PyPI version](https://badge.fury.io/py/sumo-rl.svg)](https://badge.fury.io/py/sumo-rl)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/LucasAlegre/sumo-rl/blob/main/LICENSE)

# SUMO-RL: Work Zone Traffic Signal Control with DQN

<!-- start intro -->

SUMO-RL provides a simple interface to instantiate Reinforcement Learning (RL) environments with [SUMO](https://github.com/eclipse/sumo) for Traffic Signal Control.
This project extends the original SUMO-RL framework with a **multi-model comparison pipeline** that evaluates trained DQN agents against SUMO's default fixed-timing controller in a **work zone / construction scenario**. It measures five key metrics — waiting time, queue length, average speed, throughput, and TTC (Time-To-Collision) safety conflicts — and generates Pareto front visualizations to support multi-objective model selection.

<img src="expected plots/flowchart.png" align="center" width="80%"/>

Goals of this repository:
- Provide a simple interface to work with Reinforcement Learning for Traffic Signal Control using SUMO
- Support Multiagent RL
- Compatibility with `gymnasium.Env` and popular RL libraries such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [RLlib](https://docs.ray.io/en/main/rllib.html)
- Easy customisation: state and reward definitions are easily modifiable
- **Evaluate and compare multiple DQN Pareto-optimal agents against the SUMO default controller**
- **Multi-objective analysis: efficiency (waiting, queue, speed, throughput) vs. safety (TTC conflicts)**

<img src="expected plots/workZone.png" align="center" width="70%"/>

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

Clone this Repository:
```bash
git clone https://github.com/IsraelAfriyie-dev/SUMO-REINFORCEMENT-LEARNING-FOR-WORKZONE-TRAFFIC-CONDITION
cd SUMO-REINFORCEMENT-LEARNING-FOR-WORKZONE-TRAFFIC-CONDITION
pip install -e .
```

### Install additional dependencies for the comparison pipeline:

```bash
pip install stable-baselines3 matplotlib numpy
```

<!-- end install -->

---

## Observations, Action and Rewards

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


### Action

<!-- start Action -->

The agent selects from a discrete set of traffic signal phases at the intersection:
```python
A = {0, 1, 2, ..., Np-1}
```
where Np is the total number of feasible signal phases.
<!-- end action -->



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

The script [`compare_models.py`](experiments/compare_models.py) evaluates a set of pre-trained DQN models against SUMO's default fixed-timing controller for some measured metrics across multiple episodes and seeds.

### Measured Metrics

| Metric | Description |
|---|---|
| **Waiting Time (s)** | Average cumulative waiting time across all controlled lanes |
| **Queue Length** | Average number of halted vehicles per lane (speed < 0.1 m/s) |
| **Average Speed (m/s)** | Mean vehicle speed across all controlled lanes |
| **Throughput** | Number of vehicles that successfully completed their trip |
| **TTC Conflicts** | Number of vehicles with a Time-To-Collision below 1.5 seconds (safety proxy) |

<p align="center">
<img src="expected plots/dqn_work_zone_results.png" align="center" width="50%"/>
</p>


### Configuration

```python
SIMULATION_SECONDS = 200   # Total simulation time per episode
DELTA_TIME         = 5     # Agent decision interval (seconds)
EPISODE_SEEDS      = [11, 22, 33]  # Random seeds for reproducibility
CONTROLLED_TLS_ID  = "TL1"         # Traffic light ID in your network
```

### Plotting Results
<p align="center">
<img src="expected plots/reward plots.png" align="center" width="60%"/>
</p>

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

### Pareto Multi-Objective Optimization:
The reinforcement learning framework trains multiple policies using different reward weight combinations to explore trade-offs between competing objectives: waiting time, queue length, and safety (TTC conflicts).

<p align="center">
<img src="expected plots/3D pareto.png" align="center" width="60%"/>
</p>

---



