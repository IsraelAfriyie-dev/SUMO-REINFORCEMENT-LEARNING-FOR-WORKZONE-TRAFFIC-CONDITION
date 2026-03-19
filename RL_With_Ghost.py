

import os
import sys
import csv
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


GHOST_CONFIG = {
    "edge": "E#9",          # Edge where work zone is located
    "lane": 1,              # Lane to block (lane 1)
    "pos": 320.0,           # Position on edge (meters)
    "length": 60.0,         # Length of work zone (meters)
    "duration": 999999.0,   # Keep ghost for entire training
    
    # Behavioral zones (EXACT from TraCI simulation)
    "queue_start": 200.0,   # Queue begins 200m before ghost
    "merge_start": 170.0,   # Merge begins 170m before ghost
    "force_start": 20.0,    # Panic merge 20m before ghost
    
    # Speed limits in zones (EXACT from TraCI simulation)
    "queue_speed": 1.0,     # Speed in queue zone (m/s)
    "merge_speed": 2.5,     # Speed in merge zone (m/s)
    "downstream_speed": 1.7,# Speed in downstream zone (m/s)
    "downstream_length": 90.0, # Downstream congestion length (m)
}



def get_queue_tail_pos(sumo, edge_id, lane_index, rear_pos, speed_thresh=1.0):
    """
    Queue tail position (m) on a lane:
    the most-upstream vehicle with speed < speed_thresh and position < rear_pos.
    
    EXACT from TraCI simulation - adapted for sumo-rl
    """
    lane_id = f"{edge_id}_{lane_index}"
    try:
        vids = sumo.lane.getLastStepVehicleIDs(lane_id)
    except:
        return None

    tail_pos = None
    for vid in vids:
        if vid == "ghost":  # Skip ghost vehicle itself
            continue
        try:
            p = sumo.vehicle.getLanePosition(vid)
            v = sumo.vehicle.getSpeed(vid)
            # Vehicle is slow and behind ghost
            if p < rear_pos and v < speed_thresh:
                if tail_pos is None or p < tail_pos:
                    tail_pos = p
        except:
            pass

    return tail_pos



def my_reward_fn_with_ttc_and_queue(traffic_signal):
    """
    Custom reward with TTC safety + work zone queue awareness
    """
    
    # Normal components
    waiting = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    queue = traffic_signal.get_total_queued()
    speed = traffic_signal.get_average_speed()
    pressure = traffic_signal.get_pressure()
    
    # TTC using internal TraCI
    ttc_conflicts = 0
    try:
        sumo = traffic_signal.sumo
        vehicle_ids = sumo.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            try:
                min_ttc = sumo.vehicle.getParameter(veh_id, "device.ssm.minTTC")
                if min_ttc != "" and min_ttc != "NA":
                    if float(min_ttc) < 1.5:
                        ttc_conflicts += 1
            except Exception:
                continue
    except Exception:
        ttc_conflicts = 0
    
    # Work zone queue length (optional additional penalty)
    queue_length = 0
    try:
        sumo = traffic_signal.sumo
        rear_pos = GHOST_CONFIG["pos"] - GHOST_CONFIG["length"]
        tail_pos = get_queue_tail_pos(
            sumo, 
            GHOST_CONFIG["edge"], 
            GHOST_CONFIG["lane"], 
            rear_pos
        )
        if tail_pos is not None:
            queue_length = rear_pos - tail_pos  # Queue length in meters
    except:
        pass
    
    # Normalize
    normalized_waiting = waiting / 100.0
    normalized_queue = min(queue / 50.0, 1.0)
    normalized_pressure = max(min(pressure / 20.0, 1.0), -1.0)
    normalized_ttc = min(ttc_conflicts / 10.0, 1.0)
    normalized_queue_length = min(queue_length / 200.0, 1.0)  # 0-200m queue
    
    # Combined reward (with queue length penalty)
    reward = (
        -0.3 * normalized_waiting +        # Delay penalty
        -0.20 * normalized_queue +          # Queue count penalty
        +1.60 * speed +                     # Speed bonus
        -0.10 * normalized_pressure +       # Pressure penalty
        -0.40 * normalized_ttc +            # TTC safety penalty
        -0.15 * normalized_queue_length     # Work zone queue penalty
    )
    
    return reward


class WorkZoneSumoEnv(SumoEnvironment):
   
    def __init__(self, 
                 state_file='saved_state.xml',
                 warmup_steps=120,
                 restart_every=50,
                 enable_data_collection=True,
                 csv_output_dir='./training_data/',
                 **kwargs):
        
        self.state_file = state_file
        self.warmup_steps = warmup_steps
        self.restart_every = restart_every
        self.enable_data_collection = enable_data_collection
        self.csv_output_dir = csv_output_dir
        
        self._warmed_up = False
        self._reset_count = 0
        self.crash_count = 0
        self._ghost_created = False
        
        # CSV file handles
        self.csv_file = None
        self.csv_writer = None
        self._csv_closed = False
        
        # Tracking for work zone behavior (EXACT from TraCI simulation)
        self.slowed_for_queue = set()
        self.issued_merge_cmd = set()
        self.pinned_after_merge = set()
        self.slowed_downstream = set()
        
        super().__init__(**kwargs)
    
    def _setup_csv_logging(self):
        """Initialize CSV file for trajectory logging"""
        if not self.enable_data_collection:
            return
        
        os.makedirs(self.csv_output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.csv_output_dir, f"trajectories_{timestamp}.csv")
        
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header (EXACT from TraCI simulation)
        self.csv_writer.writerow([
            'episode',
            'time_s',
            'vehicle_id',
            'position_m',
            'speed_mps',
            'lane',
            'tail_pos_m',
            'rear_pos_m',
            'queue_length_m',
            'ttc_conflicts'
        ])
        
        print(f"📊 CSV logging enabled: {csv_path}")
    
    def _create_ghost_vehicle(self):
        """Create ghost vehicle (EXACT from TraCI simulation)"""
        if self._ghost_created:
            return
        
        try:
            sumo = self.sumo
            
            # Create ghost vehicle type (EXACT from TraCI simulation)
            if "ghost" not in sumo.vehicletype.getIDList():
                sumo.vehicletype.copy("DEFAULT_VEHTYPE", "ghost")
                sumo.vehicletype.setMaxSpeed("ghost", 0.01)  # Almost stationary
                sumo.vehicletype.setColor("ghost", (0, 0, 0, 255))  # Black
                sumo.vehicletype.setLength("ghost", GHOST_CONFIG["length"])  # 60.0m
                sumo.vehicletype.setWidth("ghost", 2.6)
                sumo.vehicletype.setMinGap("ghost", 0.5)
            
            # Create ghost route
            if "ghostRoute" not in sumo.route.getIDList():
                sumo.route.add("ghostRoute", [GHOST_CONFIG["edge"]])
            
            # Add ghost vehicle
            if "ghost" not in sumo.vehicle.getIDList():
                sumo.vehicle.add(
                    vehID="ghost",
                    routeID="ghostRoute",
                    typeID="ghost",
                    depart=0,
                    departLane=GHOST_CONFIG["lane"],
                    departPos=str(GHOST_CONFIG["pos"]),
                    departSpeed="0"
                )
                
                # Make ghost stop permanently
                sumo.vehicle.setStop(
                    "ghost",
                    edgeID=GHOST_CONFIG["edge"],
                    pos=GHOST_CONFIG["pos"],
                    laneIndex=GHOST_CONFIG["lane"],
                    duration=GHOST_CONFIG["duration"]
                )
                
                self._ghost_created = True
                print(f"🚧 Ghost vehicle created at {GHOST_CONFIG['edge']} "
                      f"lane {GHOST_CONFIG['lane']}, pos {GHOST_CONFIG['pos']}m")
        
        except Exception as e:
            print(f"⚠️ Could not create ghost vehicle: {e}")
    
    def _ensure_ghost_exists(self):
        """Verify ghost vehicle exists, recreate if missing"""
        try:
            if "ghost" not in self.sumo.vehicle.getIDList():
                # Silent recreation (less verbose)
                self._ghost_created = False
                self._create_ghost_vehicle()
            else:
                self._ghost_created = True
        except Exception as e:
            self._ghost_created = False
            self._create_ghost_vehicle()
    
    def _control_work_zone_behavior(self):
        
        try:
            sumo = self.sumo
            edge_id = GHOST_CONFIG["edge"]
            ghost_pos = GHOST_CONFIG["pos"]
            ghost_len = GHOST_CONFIG["length"]
            rear_pos = ghost_pos - ghost_len
            
            vehicles = sumo.edge.getLastStepVehicleIDs(edge_id)
            
            for vid in vehicles:
                if vid == "ghost":
                    continue
                
                try:
                    lane = sumo.vehicle.getLaneIndex(vid)
                    pos = sumo.vehicle.getLanePosition(vid)
                    dist = ghost_pos - pos  # Distance to ghost front
                    
                    # Keep vehicles on lane 1 after passing the blockage (EXACT from TraCI)
                    if lane == 1 and pos >= ghost_pos and vid not in self.pinned_after_merge:
                        try:
                            sumo.vehicle.changeLane(vid, 1, 99999)
                            self.pinned_after_merge.add(vid)
                        except:
                            pass
                    
                    # Downstream congestion on lane 1 after blockage (EXACT from TraCI)
                    if (lane == 1 and ghost_pos <= pos <= 
                        ghost_pos + GHOST_CONFIG["downstream_length"]):
                        if vid not in self.slowed_downstream:
                            try:
                                sumo.vehicle.slowDown(vid, GHOST_CONFIG["downstream_speed"], 3.0)
                                self.slowed_downstream.add(vid)
                            except:
                                pass
                    
                    if dist <= 0:  # Vehicle has passed ghost
                        continue
                    
                    # ✅ CORRECT: Queue formation on lane 1 (blocked lane)
                    if (lane == GHOST_CONFIG["lane"] and 
                        dist <= GHOST_CONFIG["queue_start"] and 
                        vid not in self.slowed_for_queue):
                        try:
                            sumo.vehicle.setLaneChangeMode(vid, 0)  # No lane changes
                            sumo.vehicle.slowDown(vid, GHOST_CONFIG["queue_speed"], 4.0)
                            self.slowed_for_queue.add(vid)
                        except:
                            pass
                    
                    # Smooth merge command (adapted for LC2013 compatibility)
                    if (lane == GHOST_CONFIG["lane"] and 
                        dist <= GHOST_CONFIG["merge_start"] and 
                        vid not in self.issued_merge_cmd):
                        try:
                            sumo.vehicle.setLaneChangeMode(vid, 1621)
                            sumo.vehicle.setParameter(vid, "laneChangeModel.lcAssertive", "0.8")
                            # lcImpatience removed - not supported by LC2013 model
                            sumo.vehicle.setParameter(vid, "laneChangeModel.lcCooperative", "1.0")
                            
                            sumo.vehicle.changeLane(vid, 1, 8.0)
                            sumo.vehicle.slowDown(vid, GHOST_CONFIG["merge_speed"], 3.0)
                            
                            self.issued_merge_cmd.add(vid)
                        except:
                            pass
                    
                    # Panic merge (EXACT from TraCI)
                    if (lane == GHOST_CONFIG["lane"] and 
                        dist <= GHOST_CONFIG["force_start"] and 
                        vid not in self.issued_merge_cmd):
                        try:
                            sumo.vehicle.changeLane(vid, 1, 3.0)  # EXACT from TraCI
                            self.issued_merge_cmd.add(vid)
                        except:
                            pass
                
                except:
                    continue
        
        except Exception as e:
            pass  # Silently continue if error
    
    def _collect_trajectory_data(self):
        """Collect vehicle trajectory data to CSV (EXACT from TraCI simulation)"""
        if not self.enable_data_collection or self.csv_writer is None:
            return
        
        try:
            sumo = self.sumo
            sim_time = sumo.simulation.getTime()
            edge_id = GHOST_CONFIG["edge"]
            rear_pos = GHOST_CONFIG["pos"] - GHOST_CONFIG["length"]
            
            # Get queue tail position (EXACT from TraCI simulation)
            tail_pos = get_queue_tail_pos(
                sumo, edge_id, GHOST_CONFIG["lane"], rear_pos
            )
            queue_length = (rear_pos - tail_pos) if tail_pos is not None else 0.0
            
            # Count TTC conflicts
            ttc_conflicts = 0
            try:
                vehicle_ids = sumo.vehicle.getIDList()
                for veh_id in vehicle_ids:
                    try:
                        min_ttc = sumo.vehicle.getParameter(veh_id, "device.ssm.minTTC")
                        if min_ttc != "" and min_ttc != "NA":
                            if float(min_ttc) < 1.5:
                                ttc_conflicts += 1
                    except:
                        continue
            except:
                pass
            
            # Record all vehicles on work zone edge (EXACT from TraCI simulation)
            vehicles = sumo.edge.getLastStepVehicleIDs(edge_id)
            for vid in vehicles:
                if vid == "ghost":
                    continue
                try:
                    lane = sumo.vehicle.getLaneIndex(vid)
                    pos = sumo.vehicle.getLanePosition(vid)
                    speed = sumo.vehicle.getSpeed(vid)
                    
                    # Write row (enhanced with episode and ttc_conflicts)
                    self.csv_writer.writerow([
                        self._reset_count,  # episode number
                        f"{sim_time:.2f}",
                        vid,
                        f"{pos:.3f}",
                        f"{speed:.3f}",
                        lane,
                        f"{tail_pos:.3f}" if tail_pos is not None else "",
                        f"{rear_pos:.3f}",
                        f"{queue_length:.3f}",
                        ttc_conflicts
                    ])
                except:
                    pass
        
        except Exception as e:
            pass  # Silently continue
    
    def _start_simulation(self):
        """Start simulation with ghost vehicle"""
        super()._start_simulation()
        
        # Create ghost vehicle
        self._create_ghost_vehicle()
        
        # Setup CSV logging (once)
        if self.csv_file is None:
            self._setup_csv_logging()
        
        # Warmup
        if not self._warmed_up and self.sumo is not None:
            self._do_warmup()
            self._warmed_up = True
    
    def _do_warmup(self):
        """Warmup phase with ghost vehicle and work zone behavior"""
        print(f"\n{'='*70}")
        print(f"🔥 WARMUP PHASE - Stabilizing Traffic with Work Zone")
        print(f"{'='*70}")
        print(f"Running {self.warmup_steps} steps...")
        
        for i in range(self.warmup_steps):
            try:
                # Apply work zone behavior
                self._control_work_zone_behavior()
                
                # Step simulation
                self.sumo.simulationStep()
                
                # Collect data (even during warmup)
                if i % 5 == 0:  # Every 5 steps to reduce overhead
                    self._collect_trajectory_data()
                
                if i % 20 == 0 or i == self.warmup_steps - 1:
                    vehicles = len(self.sumo.vehicle.getIDList())
                    print(f"  Step {i:3d}/{self.warmup_steps} - Vehicles: {vehicles}")
            except Exception as e:
                print(f"⚠️ Warmup failed at step {i}: {e}")
                return
        
        # Save state (includes ghost vehicle)
        try:
            self.sumo.simulation.saveState(self.state_file)
            print(f"\n✅ State saved with ghost vehicle: {self.state_file}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"⚠️ Could not save state: {e}")
    
    def reset(self, **kwargs):
        """Reset with work zone behavior tracking reset"""
        self._reset_count += 1
        
        # Reset behavior tracking sets (EXACT from TraCI simulation)
        self.slowed_for_queue.clear()
        self.issued_merge_cmd.clear()
        self.pinned_after_merge.clear()
        self.slowed_downstream.clear()
        
        # Periodic full restart
        if self._reset_count % self.restart_every == 0:
            print(f"\n🔄 Periodic restart (episode {self._reset_count})")
            self._ghost_created = False  # Will recreate ghost
            return self._hard_reset(**kwargs)
        
        # Try normal reset
        try:
            result = super().reset(**kwargs)
            # Verify ghost after every reset
            self._ensure_ghost_exists()
            return result
        except Exception as e:
            print(f"\n⚠️ Reset failed: {e}")
            self.crash_count += 1
            self._ghost_created = False
            return self._hard_reset(**kwargs)
    
    def _hard_reset(self, **kwargs):
        """Complete restart (CSV stays open!)"""
        try:
            if self.sumo is not None:
                # Close SUMO but NOT CSV
                try:
                    import traci
                    traci.close()
                except:
                    pass
        except:
            pass
        
        self._warmed_up = False
        self._ghost_created = False
        self._start_simulation()
        
        return super().reset(**kwargs)
    
    def _sumo_reset(self):
        """Reset with state loading"""
        if self.sumo is None:
            self._start_simulation()
            return
        
        if os.path.exists(self.state_file):
            try:
                self.sumo.simulation.loadState(self.state_file)
                
                # Always verify ghost exists after state load
                self._ensure_ghost_exists()
                return
                
            except Exception as e:
                print(f"  (State load failed: {e})")
        
        super()._sumo_reset()
        self._create_ghost_vehicle()
    
    def step(self, action):
        """Step with work zone behavior control and data collection"""
        try:
            # Apply work zone behavior BEFORE step (EXACT from TraCI simulation)
            self._control_work_zone_behavior()
            
            # Normal RL step
            result = super().step(action)
            
            # Collect trajectory data AFTER step
            self._collect_trajectory_data()
            
            return result
            
        except Exception as e:
            print(f"\n⚠️ Step crashed: {e}")
            self.crash_count += 1
            obs = self.observation_space.sample()
            return obs, -10.0, True, True, {'crash': True}
    
    def close(self):
        """Close environment and CSV file (only once)"""
        # Only close CSV if not already closed
        if self.csv_file is not None and not self._csv_closed:
            try:
                self.csv_file.flush()  # Ensure all data written
                self.csv_file.close()
                self._csv_closed = True
                print(f"📁 CSV file closed successfully")
            except Exception as e:
                print(f"⚠️ Error closing CSV: {e}")
            finally:
                self.csv_file = None
                self.csv_writer = None
        
        # Close SUMO
        try:
            super().close()
        except:
            pass
    
    def __del__(self):
        """Destructor - ensure CSV is closed"""
        if self.csv_file is not None and not self._csv_closed:
            try:
                self.csv_file.close()
                self._csv_closed = True
            except:
                pass


# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("="*70)
    print("🚦 RL TRAINING WITH WORK ZONE + DATA COLLECTION")
    print("="*70)
    print("✅ Ghost vehicle (work zone) enabled")
    print("✅ SSM device (TTC tracking) enabled")
    print("✅ CSV trajectory logging enabled")
    print("✅ Custom reward with safety")
    print("✅ 100% EXACT TraCI work zone behavior")
    print("="*70 + "\n")
    
    # SSM configuration
    # SSM configuration WITH lateral resolution (CRITICAL for work zone!)
    SSM_AND_PHYSICS = (
        "--device.ssm.probability 1.0 "
        "--device.ssm.measures TTC "
        "--collision.action remove "
        "--no-step-log "
        "--lateral-resolution 0.8 "  # ← CRITICAL: Enables lateral movement for merging!
        "--step-length 0.2"           # ← Match TraCI simulation physics
    )
    
    # Create environment with work zone
    env = WorkZoneSumoEnv(
        net_file='net.net.xml',
        route_file='rou.route.xml',
        
        use_gui=False,
        single_agent=True,
        num_seconds=120,
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=50,
        
        # Custom reward with work zone awareness
        reward_fn=my_reward_fn_with_ttc_and_queue,
        
        # SSM + Physics configuration
        time_to_teleport=-1,
        max_depart_delay=-1,
        additional_sumo_cmd=SSM_AND_PHYSICS,  # ← Changed to include physics
        
        # Work zone environment features
        state_file='workzone_state.xml',
        warmup_steps=120,
        restart_every=50,
        enable_data_collection=True,
        csv_output_dir='./training_data/',
    )
    
    print(f"✅ Work zone environment created")
    print()
    print(f"{'='*70}")
    print(f"🚧 WORK ZONE CONFIGURATION")
    print(f"{'='*70}")
    print(f"   Edge: {GHOST_CONFIG['edge']}")
    print(f"   Lane: {GHOST_CONFIG['lane']} (blocked)")
    print(f"   Position: {GHOST_CONFIG['pos']}m")
    print(f"   Length: {GHOST_CONFIG['length']}m")
    print(f"   Queue zone: {GHOST_CONFIG['queue_start']}m before")
    print(f"   Merge zone: {GHOST_CONFIG['merge_start']}m before")
    print(f"   Force merge: {GHOST_CONFIG['force_start']}m before")
    print(f"   Downstream: {GHOST_CONFIG['downstream_length']}m after")
    print(f"{'='*70}\n")
    
    print(f"{'='*70}")
    print(f"📊 TRAFFIC LIGHT CONFIGURATION")
    print(f"{'='*70}")
    print(f"🚦 Traffic signals: {env.ts_ids}")
    
    if env.single_agent:
        controlled = env.ts_ids[0] if env.ts_ids else "Unknown"
        print(f"\n🤖 AI CONTROL:")
        print(f"   ✅ {controlled} - Controlled by DQN")
        print(f"   📊 Reward: Custom with TTC + Work Zone Queue")
    
    print(f"\n🎯 Reward Components:")
    print(f"   -0.35 × waiting_time/100")
    print(f"   -0.20 × queue_count/50")
    print(f"   +1.60 × speed")
    print(f"   -0.10 × pressure/20")
    print(f"   -0.40 × TTC_conflicts/10  ← SAFETY!")
    print(f"   -0.15 × queue_length/200  ← WORK ZONE!")
    print(f"{'='*70}\n")
    
    # Setup directories
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./training_data/", exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='dqn_workzone',
        verbose=1
    )
    
    # Create DQN model
    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=1e-3,
        learning_starts=0,
        buffer_size=50000,
        batch_size=32,
        train_freq=1,
        target_update_interval=500,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./logs/",
    )
    
    print("="*70)
    print("🚀 STARTING TRAINING WITH WORK ZONE")
    print("="*70)
    print("Reward: Custom with TTC + Work Zone Queue")
    print("TensorBoard: tensorboard --logdir ./logs/")
    print("="*70 + "\n")
    
    # Train
    try:
        model.learn(
            total_timesteps=300_000,
            callback=checkpoint_callback,
            log_interval=10,
            progress_bar=True,
            tb_log_name="DQN_WorkZone"
        )
        
        model.save("models/dqn_workzone_final")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)
        print(f"   Model: models/dqn_workzone_final.zip")
        print(f"   Episodes: {env._reset_count}")
        print(f"   Crashes: {env.crash_count}")
        print(f"   Crash rate: {env.crash_count/max(env._reset_count, 1)*100:.1f}%")
        print(f"   CSV data: ./training_data/")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        model.save("models/dqn_workzone_interrupted")
        print(f"   Episodes: {env._reset_count}")
        print(f"   Crashes: {env.crash_count}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        model.save("models/dqn_workzone_error")
        
    finally:
        print("\nCleaning up...")
        env.close()
        print("✅ Done")

