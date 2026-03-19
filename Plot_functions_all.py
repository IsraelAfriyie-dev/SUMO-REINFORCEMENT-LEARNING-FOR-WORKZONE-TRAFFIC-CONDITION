import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

csv_file = "vehicle_trajectories2.csv"  # Your trajectory CSV file
QUEUE_SPEED_THRESHOLD = 5.0            # m/s - vehicles slower than this are queuing

print("="*70)
print("📊 AVERAGED TRAJECTORY ANALYSIS ACROSS ALL EPISODES")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

df = pd.read_csv(csv_file)

print(f"\n✅ Loaded data from: {csv_file}")
print(f"   Total rows: {len(df):,}")
print(f"   Total vehicles: {df['vehicle_id'].nunique()}")
print(f"   Total episodes: {df['episode'].nunique()}")
print(f"   Time range per episode: {df.groupby('episode')['time_s'].min().mean():.1f}s to {df.groupby('episode')['time_s'].max().mean():.1f}s")

# ═══════════════════════════════════════════════════════════════════
# FILTER OUT STATIONARY VEHICLES GLOBALLY
# ═══════════════════════════════════════════════════════════════════

vehicle_max_speeds = df.groupby('vehicle_id')['speed_mps'].max()
vehicles_with_movement = vehicle_max_speeds[vehicle_max_speeds > QUEUE_SPEED_THRESHOLD].index.tolist()

print(f"\n🚗 Filtering vehicles:")
print(f"   Total unique vehicles: {df['vehicle_id'].nunique()}")
print(f"   Moving vehicles (speed > {QUEUE_SPEED_THRESHOLD} m/s): {len(vehicles_with_movement)}")
print(f"   Filtered out: {df['vehicle_id'].nunique() - len(vehicles_with_movement)}")

df = df[df['vehicle_id'].isin(vehicles_with_movement)]

# ═══════════════════════════════════════════════════════════════════
# NORMALIZE TIME WITHIN EACH EPISODE (0% to 100%)
# ═══════════════════════════════════════════════════════════════════

print("\n🔄 Normalizing time within episodes...")

# For each episode, normalize time to 0-100%
def normalize_time(group):
    t_min = group['time_s'].min()
    t_max = group['time_s'].max()
    if t_max > t_min:
        group['time_normalized'] = (group['time_s'] - t_min) / (t_max - t_min) * 100
    else:
        group['time_normalized'] = 0
    return group

df = df.groupby('episode', group_keys=False).apply(normalize_time)

# ═══════════════════════════════════════════════════════════════════
# CALCULATE AVERAGED METRICS ACROSS ALL EPISODES
# ═══════════════════════════════════════════════════════════════════

print("📊 Calculating averaged metrics across all episodes...")

# Create time bins for averaging (0-100% in steps)
time_bins = np.linspace(0, 100, 201)  # 201 points (0%, 0.5%, 1%, ..., 100%)

# Bin the normalized time
df['time_bin'] = pd.cut(df['time_normalized'], bins=time_bins, labels=time_bins[:-1])
df['time_bin'] = df['time_bin'].astype(float)

# 1. AVERAGE POSITION
avg_position = df.groupby('time_bin')['position_m'].mean().reset_index()
std_position = df.groupby('time_bin')['position_m'].std().reset_index()

# 2. AVERAGE SPEED
avg_speed = df.groupby('time_bin')['speed_mps'].mean().reset_index()
std_speed = df.groupby('time_bin')['speed_mps'].std().reset_index()

# 3. AVERAGE QUEUE COUNT
df['is_queued'] = df['speed_mps'] < QUEUE_SPEED_THRESHOLD

# For each episode and time bin, count queuing vehicles
queue_count_per_episode = df.groupby(['episode', 'time_bin'])['is_queued'].sum().reset_index()
queue_count_per_episode.columns = ['episode', 'time_bin', 'queuing_vehicles']

# Average queue count across episodes
avg_queue = queue_count_per_episode.groupby('time_bin')['queuing_vehicles'].mean().reset_index()
std_queue = queue_count_per_episode.groupby('time_bin')['queuing_vehicles'].std().reset_index()

# 4. AVERAGE VELOCITY DEVIATION (per episode, then averaged)
velocity_dev_per_episode = df.groupby(['episode', 'time_bin'])['speed_mps'].agg(['std', 'count']).reset_index()
velocity_dev_per_episode.columns = ['episode', 'time_bin', 'velocity_deviation', 'n_vehicles']
velocity_dev_per_episode['velocity_deviation'] = velocity_dev_per_episode['velocity_deviation'].fillna(0)

# Average velocity deviation across episodes
avg_velocity_dev = velocity_dev_per_episode.groupby('time_bin')['velocity_deviation'].mean().reset_index()
std_velocity_dev = velocity_dev_per_episode.groupby('time_bin')['velocity_deviation'].std().reset_index()

print(f"✅ Calculated averages for {len(avg_position)} time points")
print(f"\n📈 Summary Statistics (Averaged Across All Episodes):")
print(f"   Mean position: {avg_position['position_m'].mean():.1f} m (±{std_position['position_m'].mean():.1f} m)")
print(f"   Mean speed: {avg_speed['speed_mps'].mean():.2f} m/s (±{std_speed['speed_mps'].mean():.2f} m/s)")
print(f"   Mean queue count: {avg_queue['queuing_vehicles'].mean():.2f} vehicles (±{std_queue['queuing_vehicles'].mean():.2f})")
print(f"   Mean velocity deviation: {avg_velocity_dev['velocity_deviation'].mean():.2f} m/s (±{std_velocity_dev['velocity_deviation'].mean():.2f} m/s)")

# ═══════════════════════════════════════════════════════════════════
# CREATE 4-PANEL AVERAGED PLOT
# ═══════════════════════════════════════════════════════════════════

print("\n🎨 Creating 4-panel averaged analysis plot...")

fig, axes = plt.subplots(4, 1, figsize=(12, 14))

# PLOT 1: AVERAGE TRAJECTORY (Position vs Normalized Time)
ax1 = axes[0]
ax1.plot(avg_position['time_bin'], avg_position['position_m'], 
         color='blue', linewidth=2.5, label='Mean position')
ax1.fill_between(avg_position['time_bin'], 
                 avg_position['position_m'] - std_position['position_m'], 
                 avg_position['position_m'] + std_position['position_m'],
                 alpha=0.3, color='blue', label='±1 std dev')

ax1.set_xlabel('Episode Progress (%)', fontsize=12)
ax1.set_ylabel('Position (m)', fontsize=12)
ax1.set_title('Plot 1: Average Trajectory Diagram (Vehicle Position vs Time)', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xlim(0, 100)

# PLOT 2: AVERAGE SPEED PROFILE
ax2 = axes[1]
ax2.plot(avg_speed['time_bin'], avg_speed['speed_mps'], 
         color='green', linewidth=2.5, label='Mean speed')
ax2.fill_between(avg_speed['time_bin'], 
                 avg_speed['speed_mps'] - std_speed['speed_mps'], 
                 avg_speed['speed_mps'] + std_speed['speed_mps'],
                 alpha=0.3, color='green', label='±1 std dev')

ax2.set_xlabel('Episode Progress (%)', fontsize=12)
ax2.set_ylabel('Speed (m/s)', fontsize=12)
ax2.set_title('Plot 2: Average Speed Profile (Vehicle Speed vs Time)', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, max(avg_speed['speed_mps'].max() * 1.2, 10))

# PLOT 3: AVERAGE QUEUING VEHICLES
ax3 = axes[2]
ax3.plot(avg_queue['time_bin'], avg_queue['queuing_vehicles'], 
         color='blue', linewidth=2.5, label=f'Mean queuing vehicles (speed < {QUEUE_SPEED_THRESHOLD} m/s)')
ax3.fill_between(avg_queue['time_bin'], 
                 avg_queue['queuing_vehicles'] - std_queue['queuing_vehicles'], 
                 avg_queue['queuing_vehicles'] + std_queue['queuing_vehicles'],
                 alpha=0.3, color='blue', label='±1 std dev')

ax3.set_xlabel('Episode Progress (%)', fontsize=12)
ax3.set_ylabel('Number of Queuing Vehicles', fontsize=12)
ax3.set_title('Plot 3: Average Queuing Vehicles Over Time', 
              fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_xlim(0, 100)
ax3.set_ylim(0, max(avg_queue['queuing_vehicles'].max() * 1.2, 1))

# PLOT 4: AVERAGE VELOCITY DEVIATION
ax4 = axes[3]
ax4.plot(avg_velocity_dev['time_bin'], avg_velocity_dev['velocity_deviation'], 
         color='darkblue', linewidth=2.5, label='Mean velocity deviation (σᵥ)')
ax4.fill_between(avg_velocity_dev['time_bin'], 
                 avg_velocity_dev['velocity_deviation'] - std_velocity_dev['velocity_deviation'], 
                 avg_velocity_dev['velocity_deviation'] + std_velocity_dev['velocity_deviation'],
                 alpha=0.3, color='darkblue', label='±1 std dev')

ax4.set_xlabel('Episode Progress (%)', fontsize=12)
ax4.set_ylabel('Velocity Deviation (m/s)', fontsize=12)
ax4.set_title('Plot 4: Average Velocity Deviation (Traffic Flow Instability)', 
              fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_xlim(0, 100)
ax4.set_ylim(0, max(avg_velocity_dev['velocity_deviation'].max() * 1.2, 2))

plt.tight_layout()
output_file = "averaged_shockwave_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ Plot saved: {output_file}")

# ═══════════════════════════════════════════════════════════════════
# SAVE AVERAGED DATA TO CSV
# ═══════════════════════════════════════════════════════════════════

averaged_data = pd.DataFrame({
    'time_percent': avg_position['time_bin'],
    'avg_position_m': avg_position['position_m'],
    'std_position_m': std_position['position_m'],
    'avg_speed_mps': avg_speed['speed_mps'],
    'std_speed_mps': std_speed['speed_mps'],
    'avg_queuing_vehicles': avg_queue['queuing_vehicles'],
    'std_queuing_vehicles': std_queue['queuing_vehicles'],
    'avg_velocity_deviation': avg_velocity_dev['velocity_deviation'],
    'std_velocity_deviation': std_velocity_dev['velocity_deviation']
})

csv_output = "averaged_metrics.csv"
averaged_data.to_csv(csv_output, index=False)
print(f"✅ Data saved: {csv_output}")

# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎉 AVERAGED ANALYSIS COMPLETE!")
print("="*70)

print(f"\n📊 Processed:")
print(f"   Episodes: {df['episode'].nunique()}")
print(f"   Vehicles: {df['vehicle_id'].nunique()}")
print(f"   Total data points: {len(df):,}")

print(f"\n📈 Key Findings (Averaged Across All Episodes):")
print(f"   Average position range: {avg_position['position_m'].min():.1f}m to {avg_position['position_m'].max():.1f}m")
print(f"   Average speed: {avg_speed['speed_mps'].mean():.2f} m/s (±{std_speed['speed_mps'].mean():.2f} m/s)")
print(f"   Average queue size: {avg_queue['queuing_vehicles'].mean():.2f} vehicles (±{std_queue['queuing_vehicles'].mean():.2f})")
print(f"   Average velocity deviation: {avg_velocity_dev['velocity_deviation'].mean():.2f} m/s (±{std_velocity_dev['velocity_deviation'].mean():.2f} m/s)")

print(f"\n📂 Output Files:")
print(f"   ✅ {output_file} - 4-panel averaged plot")
print(f"   ✅ {csv_output} - Averaged metrics data")

print("\n" + "="*70)
print("✨ The plot shows AVERAGED behavior across ALL episodes")
print("   - Shaded regions show ±1 standard deviation")
print("   - Time is normalized (0% = start, 100% = end of episode)")
print("="*70)