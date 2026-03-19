# Your current code - RL controls TL1
env = ProductionSumoEnv(
    net_file='net.net.xml',
    single_agent=True,  # TL1 controlled by RL
    reward_fn=my_reward_fn_with_ttc,
    # ...
)

model = DQN(env=env, ...)
model.learn(total_timesteps=100_000)
model.save("models/dqn_rl_trained")  # Save trained model


# evaluate_rl.py
from stable_baselines3 import DQN

# Load trained model
model = DQN.load("models/dqn_rl_trained")

# Create test environment
env = ProductionSumoEnv(
    net_file='net.net.xml',
    use_gui=False,
    single_agent=True,
    num_seconds=3600,  # 1 hour simulation
    reward_fn=my_reward_fn_with_ttc,
)

# Run evaluation
obs, info = env.reset()
total_reward = 0
total_delay = 0
total_throughput = 0

for step in range(3600 // 5):  # 1 hour / 5s decisions
    action, _ = model.predict(obs, deterministic=True)  # Use learned policy
    obs, reward, done, truncated, info = env.step(action)
    
    total_reward += reward
    total_delay += get_total_delay()  # Your metric function
    total_throughput += count_completed_vehicles()
    
    if done or truncated:
        break

print(f"RL Performance:")
print(f"  Total Reward: {total_reward}")
print(f"  Average Delay: {total_delay / step}")
print(f"  Throughput: {total_throughput} vehicles")

env.close()


# evaluate_fixed.py
import traci

# Start SUMO with ONLY fixed timing (no RL!)
sumo_cmd = [
    "sumo",
    "-c", "network.sumocfg",
    "--no-step-log",
]

traci.start(sumo_cmd)

total_delay = 0
total_throughput = 0

# Run for 1 hour
for step in range(3600):  # 3600 seconds
    traci.simulationStep()
    
    # Collect metrics
    vehicles = traci.vehicle.getIDList()
    for veh in vehicles:
        total_delay += traci.vehicle.getAccumulatedWaitingTime(veh)
    
    # Count arrived vehicles
    total_throughput += traci.simulation.getArrivedNumber()

print(f"Fixed Timing Performance:")
print(f"  Average Delay: {total_delay / 3600}")
print(f"  Throughput: {total_throughput} vehicles")

traci.close()
```

---

### **STEP 4: Compare Results**
```
╔════════════════════════════════════════════════════════════╗
║                    PERFORMANCE COMPARISON                   ║
╠════════════════════════════════════════════════════════════╣
║  Metric          │  Fixed Timing  │  RL Timing  │  Improvement ║
╠══════════════════╪════════════════╪═════════════╪══════════════╣
║  Avg Delay       │  45.2s        │  28.3s      │  -37% ✅     ║
║  Throughput      │  850 veh/hr   │  1120 veh/hr│  +32% ✅     ║
║  Avg Speed       │  8.5 m/s      │  11.2 m/s   │  +32% ✅     ║
║  TTC Conflicts   │  45           │  12         │  -73% ✅     ║
╚══════════════════╧════════════════╧═════════════╧══════════════╝

Conclusion: RL learned better timings! 🎉