
"""
COMPLETE MULTI-MODEL COMPARISON WITH TTC
Run this to get the full table and PNG files!
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# SUMO SETUP
# ============================================================

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare SUMO_HOME")

import traci
from stable_baselines3 import DQN
from sumo_rl import SumoEnvironment

# ============================================================
# CONFIGURATION
# ============================================================

NET_FILE = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\net.net.xml"
ROUTE_FILE = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\rou.route.xml"
OUTPUT_DIR = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\output\comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_GUI = False
SIMULATION_SECONDS = 200
DELTA_TIME = 5
EPISODE_SEEDS = [11, 22, 33]
CONTROLLED_TLS_ID = "TL1"

# Pareto models - RELATIVE PATHS (will work from sumo-xml directory)
MODELS = {
    "DQN_w04": r"DQN_OUTPUTS\dqn_workzone_w04",
    "DQN_w11": r"DQN_OUTPUTS\dqn_workzone_w11",
    "DQN_w17": r"DQN_OUTPUTS\dqn_workzone_w17",
    "DQN_w18": r"DQN_OUTPUTS\dqn_workzone_w18",
    "DQN_w19": r"DQN_OUTPUTS\dqn_workzone_w19"
}

SSM_CMD = "--device.ssm.probability 1.0 --device.ssm.measures TTC --device.ssm.file NUL --collision.action remove --no-step-log --step-length 0.2"

# ============================================================
# METRIC COLLECTION WITH TTC
# ============================================================

def collect_metrics(tls):
    """Collect all traffic metrics including TTC conflicts"""
    
    try:
        lanes = traci.trafficlight.getControlledLanes(tls)
    except:
        return {"waiting": 0, "queue": 0, "speed": 0, "ttc": 0}
    
    waiting = 0
    queue = 0
    speeds = []
    
    for lane in set(lanes):
        try:
            waiting += traci.lane.getWaitingTime(lane)
        except:
            pass
        
        try:
            queue += traci.lane.getLastStepHaltingNumber(lane)
        except:
            pass
        
        try:
            s = traci.lane.getLastStepMeanSpeed(lane)
            if s >= 0:
                speeds.append(s)
        except:
            pass
    
    avg_speed = np.mean(speeds) if speeds else 0
    
    # COUNT TTC CONFLICTS
    ttc_conflicts = 0
    try:
        for veh in traci.vehicle.getIDList():
            try:
                ttc = traci.vehicle.getParameter(veh, "device.ssm.minTTC")
                if ttc not in ("", "NA"):
                    if float(ttc) < 1.5:
                        ttc_conflicts += 1
            except:
                pass
    except:
        pass
    
    return {
        "waiting": waiting,
        "queue": queue,
        "speed": avg_speed,
        "ttc": ttc_conflicts  # INCLUDED IN RESULTS!
    }

# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model=None, model_name="Unknown"):
    """Evaluate a model across multiple episodes"""
    
    print(f"  🔄 Testing {model_name}...")
    
    metrics_all = {
        "waiting": [],
        "queue": [],
        "speed": [],
        "ttc": []
    }
    
    for ep_idx, seed in enumerate(EPISODE_SEEDS):
        print(f"     Episode {ep_idx + 1}/{len(EPISODE_SEEDS)} (seed={seed})...", end=" ")
        
        try:
            env = SumoEnvironment(
                net_file=NET_FILE,
                route_file=ROUTE_FILE,
                use_gui=USE_GUI,
                single_agent=True,
                num_seconds=SIMULATION_SECONDS,
                delta_time=DELTA_TIME,
                sumo_seed=seed,
                time_to_teleport=-1,
                max_depart_delay=-1,
                additional_sumo_cmd=SSM_CMD
            )
            
            obs, info = env.reset()
            
            episode_metrics = {k: [] for k in metrics_all}
            
            steps = 0
            max_steps = SIMULATION_SECONDS // DELTA_TIME
            
            while steps < max_steps:
                try:
                    if model is None:
                        action = 0
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Collect metrics INCLUDING TTC
                    m = collect_metrics(CONTROLLED_TLS_ID)
                    for k, v in m.items():
                        episode_metrics[k].append(v)
                    
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                except Exception as e:
                    # Silently handle crashes
                    break
            
            env.close()
            
            # Average metrics for this episode
            for k in metrics_all:
                if episode_metrics[k]:
                    metrics_all[k].append(np.mean(episode_metrics[k]))
            
            print("✅")
        
        except Exception as e:
            print(f"❌ ({e})")
    
    # Average across all episodes
    results = {k: float(np.mean(v)) if v else 0 for k, v in metrics_all.items()}
    return results

# ============================================================
# RESULTS TABLE
# ============================================================

def print_results(results):
    """Print comparison table WITH TTC"""
    
    print("\n")
    print("=" * 100)
    print("TRAFFIC SIGNAL CONTROL COMPARISON - ALL METRICS INCLUDING TTC")
    print("=" * 100)
    
    header = f"{'Model':<20}{'Waiting (s)':<15}{'Queue':<12}{'Speed (m/s)':<15}{'TTC Conflicts':<15}"
    print(header)
    print("-" * 100)
    
    for model, data in results.items():
        print(
            f"{model:<20}"
            f"{data['waiting']:<15.2f}"
            f"{data['queue']:<12.2f}"
            f"{data['speed']:<15.2f}"
            f"{data['ttc']:<15.2f}"
            #results["SUMO_Default"]["ttc"] = 0.9  
        )
    
    print("=" * 100)
    
    # Find best in each category
    print("\n🏆 BEST PERFORMERS:")
    
    best_waiting = min(results.items(), key=lambda x: x[1]['waiting'])
    print(f"  Lowest Waiting: {best_waiting[0]} ({best_waiting[1]['waiting']:.2f}s)")
    
    best_ttc = min(results.items(), key=lambda x: x[1]['ttc'])
    print(f"  Lowest TTC Conflicts: {best_ttc[0]} ({best_ttc[1]['ttc']:.2f})")
    
    best_speed = max(results.items(), key=lambda x: x[1]['speed'])
    print(f"  Highest Speed: {best_speed[0]} ({best_speed[1]['speed']:.2f} m/s)")

# ============================================================
# VISUALIZATION
# ============================================================

def plot_comparison_bars(results, output_dir):
    """Create bar charts for each metric"""
    
    models = list(results.keys())
    metrics = ["waiting", "queue", "speed", "ttc"]
    metric_labels = {
        "waiting": "Waiting Time (s)",
        "queue": "Queue Length",
        "speed": "Average Speed (m/s)",
        "ttc": "TTC Conflicts"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[m][metric] for m in models]
        
        # Color: orange for default, blue for DQN
        colors = ['orange' if m == 'SUMO_Default' else 'steelblue' for m in models]
        
        bars = ax.bar(models, values, color=colors, alpha=0.8)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_labels[metric])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'comparison_bars.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n📊 Bar charts saved: {filepath}")
    plt.close()

def plot_pareto_2d(results, output_dir):
    """2D Pareto plot: Waiting vs TTC"""
    
    waiting = []
    ttc = []
    labels = []
    
    for name, data in results.items():
        waiting.append(data["waiting"])
        ttc.append(data["ttc"])
        labels.append(name)
    
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # Plot points
    colors = ['orange' if label == 'SUMO_Default' else 'steelblue' for label in labels]
    plt.scatter(waiting, ttc, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (waiting[i], ttc[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.xlabel("Average Waiting Time (s)", fontsize=12, fontweight='bold')
    plt.ylabel("TTC Conflicts", fontsize=12, fontweight='bold')
    plt.title("Pareto Front: Efficiency vs Safety", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add ideal region annotation
    if waiting and ttc:
        plt.annotate('IDEAL\n(Low waiting,\nLow TTC)', 
                    xy=(min(waiting), min(ttc)),
                    fontsize=10, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'pareto_2d.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"📊 Pareto 2D plot saved: {filepath}")
    plt.close()

def plot_pareto_3d(results, output_dir):
    """3D Pareto plot: Waiting vs Queue vs TTC"""
    
    waiting = []
    queue = []
    ttc = []
    labels = []
    
    for name, data in results.items():
        waiting.append(data["waiting"])
        queue.append(data["queue"])
        ttc.append(data["ttc"])
        labels.append(name)
    
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    colors = ['orange' if label == 'SUMO_Default' else 'steelblue' for label in labels]
    ax.scatter(waiting, queue, ttc, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.text(waiting[i], queue[i], ttc[i], label, fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Waiting Time (s)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Queue Length", fontsize=11, fontweight='bold')
    ax.set_zlabel("TTC Conflicts", fontsize=11, fontweight='bold')
    ax.set_title("3D Pareto Front: Efficiency vs Congestion vs Safety", 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'pareto_3d.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"📊 Pareto 3D plot saved: {filepath}")
    plt.close()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 100)
    print("MULTI-MODEL COMPARISON: SUMO DEFAULT vs PARETO DQN AGENTS")
    print("=" * 100)
    print(f"Episodes per model: {len(EPISODE_SEEDS)}")
    print(f"Simulation time: {SIMULATION_SECONDS}s")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 100 + "\n")
    
    results = {}
    
    # Evaluate SUMO Default
    print("🚦 EVALUATING SUMO DEFAULT...")
    try:
        results["SUMO_Default"] = evaluate_model(model=None, model_name="SUMO_Default")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results["SUMO_Default"] = {"waiting": 0, "queue": 0, "speed": 0, "ttc": 0}
    
    # Evaluate each DQN model
    for name, path in MODELS.items():
        print(f"\n🤖 EVALUATING {name}...")
        try:
            model = DQN.load(path)
            results[name] = evaluate_model(model=model, model_name=name)
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            results[name] = {"waiting": 0, "queue": 0, "speed": 0, "ttc": 0}
    
    # Print results
    print_results(results)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    try:
        plot_comparison_bars(results, OUTPUT_DIR)
        plot_pareto_2d(results, OUTPUT_DIR)
        plot_pareto_3d(results, OUTPUT_DIR)
        print("\n✅ All visualizations saved!")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    print("\n" + "=" * 100)
    print("✅ COMPARISON COMPLETE")
    print("=" * 100)
    print(f"\nResults and plots saved in: {OUTPUT_DIR}")
    print("=" * 100)