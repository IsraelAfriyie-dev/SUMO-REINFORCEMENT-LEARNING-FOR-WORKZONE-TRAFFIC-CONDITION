"""
TensorBoard Log Plotter - White Background, No Grid
Reads TensorBoard logs and creates clean publication-ready plots
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# ============================================================
# CONFIGURATION
# ============================================================

# Path to your logs folder
LOG_DIR = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\logs"

# Output file name
OUTPUT_FILE = "training_plot_white_background.png"

# Plot settings
FIGURE_WIDTH = 14
FIGURE_HEIGHT = 8
DPI = 300

# ============================================================
# READ TENSORBOARD LOGS
# ============================================================

def read_tensorboard_logs(log_dir, metric_name='rollout/ep_rew_mean'):
    """
    Read TensorBoard logs and extract data
    
    Args:
        log_dir: Path to logs directory
        metric_name: Name of metric to extract
    
    Returns:
        Dictionary with {agent_name: {'steps': [...], 'values': [...]}}
    """
    data = {}
    
    # List all subdirectories (each is an agent)
    if not os.path.exists(log_dir):
        print(f"❌ Error: Log directory not found: {log_dir}")
        return data
    
    subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    
    print(f"📂 Found {len(subdirs)} training runs")
    
    for subdir in sorted(subdirs):
        log_path = os.path.join(log_dir, subdir)
        
        try:
            # Load the event file
            ea = event_accumulator.EventAccumulator(log_path)
            ea.Reload()
            
            # Check if metric exists
            available_scalars = ea.Tags().get('scalars', [])
            
            if metric_name in available_scalars:
                # Extract data
                scalar_events = ea.Scalars(metric_name)
                
                steps = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]
                
                data[subdir] = {
                    'steps': steps,
                    'values': values
                }
                
                print(f"   ✅ {subdir}: {len(steps)} data points")
            else:
                print(f"   ⚠️ {subdir}: Metric '{metric_name}' not found")
                print(f"      Available metrics: {available_scalars[:3]}...")
        
        except Exception as e:
            print(f"   ❌ {subdir}: Error reading logs - {e}")
    
    return data


# ============================================================
# CREATE PLOT
# ============================================================

def create_plot(data, output_file):
    """
    Create clean white background plot with no grid
    
    Args:
        data: Dictionary with agent data
        output_file: Output filename
    """
    if not data:
        print("❌ No data to plot!")
        return
    
    # Create figure with WHITE background
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor='white')
    ax.set_facecolor('white')
    
    # Color palette (20 distinct colors)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    # Plot each agent
    for idx, (agent_name, agent_data) in enumerate(sorted(data.items())):
        steps = agent_data['steps']
        values = agent_data['values']
        
        # Get color
        color = colors[idx % len(colors)]
        
        # Plot line
        ax.plot(steps, values, 
                color=color, 
                linewidth=2, 
                alpha=0.8, 
                label=agent_name)
    
    # ============================================================
    # STYLING - WHITE BACKGROUND, NO GRID
    # ============================================================
    
    # Axis labels
    ax.set_xlabel('Training Steps', fontsize=14, color='black', fontweight='bold')
    ax.set_ylabel('Average Episode Reward', fontsize=14, color='black', fontweight='bold')
   #ax.set_title('DQN Training Progress - Pareto Multi-Objective Optimization', 
    fontsize=16, color='black', fontweight='bold', pad=20)
    
    # Remove grid
    ax.grid(True)
    
    # Black spines (borders)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # Black tick labels
    ax.tick_params(colors='black', labelsize=11, width=1.5)
    
    # Legend
    ax.legend(loc='best', 
              frameon=True, 
              facecolor='white', 
              edgecolor='black',
              fontsize=9,
              ncol=2)  # 2 columns to fit more agents
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_file}")
    print(f"   Resolution: {FIGURE_WIDTH}x{FIGURE_HEIGHT} inches @ {DPI} DPI")
    
    # Show
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("📊 TENSORBOARD LOG PLOTTER - WHITE BACKGROUND")
    print("="*70)
    print(f"Log directory: {LOG_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*70 + "\n")
    
    # Read logs
    print("📖 Reading TensorBoard logs...")
    data = read_tensorboard_logs(LOG_DIR)
    
    if data:
        print(f"\n✅ Successfully loaded {len(data)} agents")
        
        # Create plot
        print("\n🎨 Creating plot...")
        create_plot(data, OUTPUT_FILE)
        
        print("\n" + "="*70)
        print("🎉 DONE!")
        print("="*70)
    else:
        print("\n❌ No data found. Check your log directory path.")
