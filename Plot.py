
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

csv_file = "vehicle_trajectories2.csv"   # change if needed
QUEUE_SPEED_THRESHOLD = 5.0              # m/s

print("="*70)
print("📊 EPISODE-AVERAGED SHOCKWAVE ANALYSIS")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

df = pd.read_csv(csv_file)

print(f"\n✅ Loaded data from: {csv_file}")
print(f"   Total rows: {len(df):,}")
print(f"   Episodes: {df['episode'].nunique()}")
print(f"   Vehicles: {df['vehicle_id'].nunique()}")

# ═══════════════════════════════════════════════════════════════════
# FILTER OUT STATIONARY VEHICLES
# ═══════════════════════════════════════════════════════════════════

vehicle_max_speeds = df.groupby('vehicle_id')['speed_mps'].max()
moving_vehicles = vehicle_max_speeds[
    vehicle_max_speeds > QUEUE_SPEED_THRESHOLD
].index.tolist()

df = df[df['vehicle_id'].isin(moving_vehicles)]

print(f"🚗 Vehicles after filtering: {df['vehicle_id'].nunique()}")

# ═══════════════════════════════════════════════════════════════════
# CREATE 4-PANEL EPISODE-AVERAGED SHOCKWAVE PLOT
# ═══════════════════════════════════════════════════════════════════

print("\n🎨 Generating episode-averaged shockwave plot...")

episodes = sorted(df['episode'].unique())

fig, axes = plt.subplots(4, 1, figsize=(12, 16))

for ep in episodes:

    ep_df = df[df['episode'] == ep].copy()

    # Ensure sorted by time
    ep_df = ep_df.sort_values('time_s')

    # Average position per time
    pos_time = ep_df.groupby('time_s')['position_m'].mean()

    # Average speed per time
    speed_time = ep_df.groupby('time_s')['speed_mps'].mean()

    # Queue count per time
    ep_df['is_queued'] = ep_df['speed_mps'] < QUEUE_SPEED_THRESHOLD
    queue_time = ep_df.groupby('time_s')['is_queued'].sum()

    # Velocity deviation per time
    vel_dev_time = ep_df.groupby('time_s')['speed_mps'].std().fillna(0)

    # Plot curves (each episode = one curve)
    axes[0].plot(pos_time.index, pos_time.values, alpha=0.4)
    axes[1].plot(speed_time.index, speed_time.values, alpha=0.4)
    axes[2].plot(queue_time.index, queue_time.values, alpha=0.4)
    axes[3].plot(vel_dev_time.index, vel_dev_time.values, alpha=0.4)

# ═══════════════════════════════════════════════════════════════════
# FORMAT PLOTS
# ═══════════════════════════════════════════════════════════════════

axes[0].set_title("Episode-Averaged Trajectory Diagram", fontweight='bold')
axes[0].set_ylabel("Average Position (m)")
axes[0].set_xlabel("Time (s)")
axes[0].grid(True)

axes[1].set_title("Episode-Averaged Speed Profiles", fontweight='bold')
axes[1].set_ylabel("Average Speed (m/s)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True)

axes[2].set_title("Episode-Averaged Queuing Vehicles", fontweight='bold')
axes[2].set_ylabel("Number of Queuing Vehicles")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True)

axes[3].set_title("Episode-Averaged Velocity Deviation Instability)", fontweight='bold')
axes[3].set_ylabel("Velocity Deviation (m/s)")
axes[3].set_xlabel("Time (s)")
axes[3].grid(True)

plt.tight_layout()
plt.savefig("episode_averaged_shockwave.png", dpi=300)
plt.close(fig)

print("✅ Saved: episode_averaged_shockwave.png")
print("="*70)
print("🎉 DONE")
print("="*70)