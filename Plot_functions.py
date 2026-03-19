import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = "vehicle_trajectories.csv"

df = pd.read_csv(csv_file)

# ===== FILTER OUT VEHICLES WITH LOW SPEEDS =====
vehicle_max_speeds = df.groupby('vehicle_id')['speed_mps'].max()
vehicles_with_movement = vehicle_max_speeds[vehicle_max_speeds > 5].index.tolist()

print(f"📊 Total vehicles in data: {df['vehicle_id'].nunique()}")
print(f"🚗 Vehicles with speed > 5 m/s: {len(vehicles_with_movement)}")
print(f"🚫 Vehicles filtered out (speed ≤ 5 m/s): {df['vehicle_id'].nunique() - len(vehicles_with_movement)}")

df = df[df['vehicle_id'].isin(vehicles_with_movement)]
vehicle_ids = df['vehicle_id'].unique()
n_vehicles = len(vehicle_ids)

# Generate colors
colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
colors2 = plt.cm.tab20b(np.linspace(0, 1, 20))
colors3 = plt.cm.tab20c(np.linspace(0, 1, 20))
colors4 = plt.cm.Set3(np.linspace(0, 1, 12))
colors5 = plt.cm.Paired(np.linspace(0, 1, 12))
all_colors = np.vstack([colors1, colors2, colors3, colors4, colors5])

if n_vehicles > len(all_colors):
    extra_colors = plt.cm.rainbow(np.linspace(0, 1, n_vehicles - len(all_colors)))
    all_colors = np.vstack([all_colors, extra_colors])

colors = all_colors[:n_vehicles]
np.random.seed(42)
np.random.shuffle(colors)

print(f"🎨 Plotting {n_vehicles} vehicles with {len(colors)} colors")

# ===== CALCULATE QUEUE COUNT AT EACH TIME STEP =====
print("\n📊 Calculating queue count over time...")
QUEUE_SPEED_THRESHOLD = 5.0  # m/s - vehicles slower than this are considered queuing

# Mark vehicles as queuing
df['is_queued'] = df['speed_mps'] < QUEUE_SPEED_THRESHOLD

# Count how many vehicles are queuing at each time step
queue_count = df.groupby('time_s')['is_queued'].sum().reset_index()
queue_count.columns = ['time_s', 'queuing_vehicles']

print(f"✅ Queue count calculated for {len(queue_count)} time steps")
print(f"   Max queuing vehicles: {queue_count['queuing_vehicles'].max():.0f}")
print(f"   Mean queuing vehicles: {queue_count['queuing_vehicles'].mean():.2f}")

# ===== CALCULATE VELOCITY DEVIATION =====
print("\n📊 Calculating velocity deviation...")
velocity_deviation = df.groupby('time_s')['speed_mps'].agg(['mean', 'std', 'count']).reset_index()
velocity_deviation.columns = ['time_s', 'mean_speed', 'velocity_deviation', 'n_vehicles']
velocity_deviation['velocity_deviation'] = velocity_deviation['velocity_deviation'].fillna(0)
print(f"✅ Calculated velocity deviation for {len(velocity_deviation)} time steps")
print(f"   Max deviation: {velocity_deviation['velocity_deviation'].max():.2f} m/s")
print(f"   Mean deviation: {velocity_deviation['velocity_deviation'].mean():.2f} m/s")

# Common parameters
position_range = df['position_m'].max() - df['position_m'].min()
position_center = (df['position_m'].max() + df['position_m'].min()) / 2
time_range = df['time_s'].max() - df['time_s'].min()
speed_max = df['speed_mps'].quantile(0.95)
queue_count_max = queue_count['queuing_vehicles'].max()
deviation_max = velocity_deviation['velocity_deviation'].quantile(0.95)

# ===============================================
# FIGURE 1: 2-PANEL (COMMENTED OUT)
# ===============================================
# print("\n🎨 Creating Figure 1 (2-panel)...")
# fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# for idx, vid in enumerate(vehicle_ids):
#     veh_data = df[df['vehicle_id'] == vid]
#     ax1.plot(veh_data['time_s'], veh_data['position_m'], 
#             color=colors[idx], alpha=0.8, linewidth=1.2)

# ax1.set_xlabel('Time (s)', fontsize=12)
# ax1.set_ylabel('Position (m)', fontsize=12)
# ax1.set_title('Trajectory Diagram: Vehicle Position vs Time', fontsize=13, fontweight='bold')
# ax1.grid(True, alpha=0.3)
# ax1.set_ylim(position_center - position_range*0.4, position_center + position_range*0.4)
# ax1.set_xlim(df['time_s'].min() + time_range*0.1, df['time_s'].max() - time_range*0.1)

# for idx, vid in enumerate(vehicle_ids):
#     veh_data = df[df['vehicle_id'] == vid]
#     ax2.plot(veh_data['time_s'], veh_data['speed_mps'], 
#             color=colors[idx], alpha=0.8, linewidth=1.2)

# ax2.set_xlabel('Time (s)', fontsize=12)
# ax2.set_ylabel('Speed (m/s)', fontsize=12)
# ax2.set_title('Speed Profiles: Vehicle Speed vs Time', fontsize=13, fontweight='bold')
# ax2.grid(True, alpha=0.3)
# ax2.set_ylim(0, speed_max * 1.1)
# ax2.set_xlim(df['time_s'].min() + time_range*0.1, df['time_s'].max() - time_range*0.1)

# plt.tight_layout()
# plt.savefig("vehicle_trajectories_and_speeds.png", dpi=300, bbox_inches='tight')
# plt.close(fig1)
# print("✅ Figure 1 (2-panel) saved as: vehicle_trajectories_and_speeds.png")

# ===============================================
# FIGURE 2: 3-PANEL (COMMENTED OUT)
# ===============================================
# print("\n🎨 Creating Figure 2 (3-panel)...")
# fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(10, 12))

# for idx, vid in enumerate(vehicle_ids):
#     veh_data = df[df['vehicle_id'] == vid]
#     ax3.plot(veh_data['time_s'], veh_data['position_m'], 
#             color=colors[idx], alpha=0.8, linewidth=1.2)

# ax3.set_xlabel('Time (s)', fontsize=12)
# ax3.set_ylabel('Position (m)', fontsize=12)
# ax3.set_title('Plot 1: Trajectory Diagram (Vehicle Position vs Time)', fontsize=13, fontweight='bold')
# ax3.grid(True, alpha=0.3)
# ax3.set_ylim(position_center - position_range*0.4, position_center + position_range*0.4)
# ax3.set_xlim(df['time_s'].min() + time_range*0.1, df['time_s'].max() - time_range*0.1)

# for idx, vid in enumerate(vehicle_ids):
#     veh_data = df[df['vehicle_id'] == vid]
#     ax4.plot(veh_data['time_s'], veh_data['speed_mps'], 
#             color=colors[idx], alpha=0.8, linewidth=1.2)

# ax4.set_xlabel('Time (s)', fontsize=12)
# ax4.set_ylabel('Speed (m/s)', fontsize=12)
# ax4.set_title('Plot 2: Speed Profiles (Vehicle Speed vs Time)', fontsize=13, fontweight='bold')
# ax4.grid(True, alpha=0.3)
# ax4.set_ylim(0, speed_max * 1.1)
# ax4.set_xlim(df['time_s'].min() + time_range*0.1, df['time_s'].max() - time_range*0.1)

# # PLOT 3: QUEUING VEHICLES COUNT OVER TIME (like your reference image)
# ax5.plot(queue_count['time_s'], queue_count['queuing_vehicles'], 
#          color='blue', linewidth=2, label=f'Queuing vehicles (speed < {QUEUE_SPEED_THRESHOLD} m/s)')

# ax5.set_xlabel('Time (s)', fontsize=12)
# ax5.set_ylabel('Number of Queuing Vehicles', fontsize=12)
# ax5.set_title('Plot 3: Queuing Vehicles Over Time', fontsize=13, fontweight='bold')
# ax5.grid(True, alpha=0.3)
# ax5.legend(fontsize=10)
# ax5.set_ylim(0, queue_count_max * 1.1)
# ax5.set_xlim(df['time_s'].min() + time_range*0.1, df['time_s'].max() - time_range*0.1)

# plt.tight_layout()
# plt.savefig("vehicle_three_panel_analysis.png", dpi=300, bbox_inches='tight')
# plt.close(fig2)
# print("✅ Figure 2 (3-panel) saved as: vehicle_three_panel_analysis.png")

# ===============================================
# FIGURE 3: 4-PANEL WITH VELOCITY DEVIATION
# ===============================================
print("\n🎨 Creating Figure 3 (4-panel with velocity deviation)...")
fig3, (ax6, ax7, ax8, ax9) = plt.subplots(4, 1, figsize=(10, 14))

for idx, vid in enumerate(vehicle_ids):
    veh_data = df[df['vehicle_id'] == vid]
    ax6.plot(veh_data['time_s'], veh_data['position_m'], 
            color=colors[idx], alpha=0.8, linewidth=1.2)

ax6.set_xlabel('Time (s)', fontsize=12)
ax6.set_ylabel('Position (m)', fontsize=12)
ax6.set_title('Plot 1: Trajectory Diagram (Vehicle Position vs Time)', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_ylim(position_center - position_range*0.4, position_center + position_range*0.4)
ax6.set_xlim(df['time_s'].min() + time_range*0.1, df['time_s'].max() - time_range*0.1)

for idx, vid in enumerate(vehicle_ids):
    veh_data = df[df['vehicle_id'] == vid]
    ax7.plot(veh_data['time_s'], veh_data['speed_mps'], 
            color=colors[idx], alpha=0.8, linewidth=1.2)

ax7.set_xlabel('Time (s)', fontsize=12)
ax7.set_ylabel('Speed (m/s)', fontsize=12)
ax7.set_title('Plot 2: Speed Profiles (Vehicle Speed vs Time)', fontsize=13, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.set_ylim(0, speed_max * 1.1)
ax7.set_xlim(df['time_s'].min() + time_range*0.1, df['time_s'].max() - time_range*0.1)

# PLOT 3: QUEUING VEHICLES COUNT OVER TIME
ax8.plot(queue_count['time_s'], queue_count['queuing_vehicles'], 
         color='blue', linewidth=2, label=f'Queuing vehicles (speed < {QUEUE_SPEED_THRESHOLD} m/s)')

ax8.set_xlabel('Time (s)', fontsize=12)
ax8.set_ylabel('Number of Queuing Vehicles', fontsize=12)
ax8.set_title('Plot 3: Queuing Vehicles Over Time', fontsize=13, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=10)
ax8.set_ylim(0, queue_count_max * 1.1)
ax8.set_xlim(df['time_s'].min() + time_range*0.1, df['time_s'].max() - time_range*0.1)

# PLOT 4: VELOCITY DEVIATION
ax9.plot(velocity_deviation['time_s'], velocity_deviation['velocity_deviation'], 
         color='darkblue', linewidth=2.5, label='Velocity Deviation (σᵥ)')

ax9.set_xlabel('Time (s)', fontsize=12)
ax9.set_ylabel('Velocity Deviation (m/s)', fontsize=12)
ax9.set_title('Plot 4: Velocity Deviation (Traffic Flow Instability)', fontsize=13, fontweight='bold')
ax9.grid(True, alpha=0.3)
ax9.legend(fontsize=10)
ax9.set_ylim(0, deviation_max * 1.15)
ax9.set_xlim(df['time_s'].min() + time_range*0.1, df['time_s'].max() - time_range*0.1)

plt.tight_layout()
plt.savefig("shockwave_analysis.png", dpi=300, bbox_inches='tight')
plt.close(fig3)
print("✅ Figure saved as: shockwave_analysis.png")

print("\n" + "="*70)
print("🎉 SUCCESS! SHOCKWAVE ANALYSIS FIGURE CREATED!")
print("="*70)
print("📁 File created:")
print("   ✅ shockwave_analysis.png (4-panel)")
print("\n🎯 Plot 3 (Queuing Vehicles) shows:")
print(f"   • Max queuing vehicles: {queue_count['queuing_vehicles'].max():.0f}")
print(f"   • Mean queuing vehicles: {queue_count['queuing_vehicles'].mean():.2f}")
print(f"   • Threshold: vehicles with speed < {QUEUE_SPEED_THRESHOLD} m/s")
print("\n🎯 Plot 4 (Velocity Deviation) shows shockwave instability:")
print(f"   • Max deviation: {velocity_deviation['velocity_deviation'].max():.2f} m/s")
print(f"   • Mean deviation: {velocity_deviation['velocity_deviation'].mean():.2f} m/s")
print("   • High peaks = strong shockwaves")
print("   • Low values = stable flow")
print("="*70)