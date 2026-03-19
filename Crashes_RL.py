import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


class ProductionSumoEnv(SumoEnvironment):
    
    
    def __init__(self, 
                 state_file='saved_state.xml',
                 warmup_steps=120,
                 restart_every=50,
                 **kwargs):
        
        self.state_file = state_file
        self.warmup_steps = warmup_steps
        self.restart_every = restart_every
        
        self._warmed_up = False
        self._reset_count = 0
        self.crash_count = 0
        
        # Call parent constructor
        super().__init__(**kwargs)
    
    def _start_simulation(self):
        """Start SUMO and run warmup"""
        super()._start_simulation()
        
        if not self._warmed_up and self.sumo is not None:
            self._do_warmup()
            self._warmed_up = True
    
    def _do_warmup(self):
        """Run warmup and save state"""
        print(f"\n{'='*70}")
        print(f"🔥 WARMUP PHASE - Stabilizing Traffic")
        print(f"{'='*70}")
        print(f"Running {self.warmup_steps} steps...")
        
        for i in range(self.warmup_steps):
            try:
                self.sumo.simulationStep()
                
                if i % 20 == 0 or i == self.warmup_steps - 1:
                    vehicles = len(self.sumo.vehicle.getIDList())
                    print(f"  Step {i:3d}/{self.warmup_steps} - Vehicles: {vehicles}")
                    
            except Exception as e:
                print(f"⚠️ Warmup failed at step {i}: {e}")
                return
        
        # Save state
        try:
            self.sumo.simulation.saveState(self.state_file)
            print(f"\n✅ State saved: {self.state_file}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"⚠️ Could not save state: {e}")
    
    def reset(self, **kwargs):
        """Smart reset strategy"""
        self._reset_count += 1
        
        # Periodic full restart
        if self._reset_count % self.restart_every == 0:
            print(f"\n🔄 Periodic restart (episode {self._reset_count})")
            return self._hard_reset(**kwargs)
        
        # Try normal reset
        try:
            return super().reset(**kwargs)
        except Exception as e:
            print(f"\n⚠️ Reset failed: {e}")
            print("  Performing hard reset...")
            self.crash_count += 1
            return self._hard_reset(**kwargs)
    
    def _hard_reset(self, **kwargs):
        """Complete restart"""
        try:
            if self.sumo is not None:
                self.close()
        except:
            pass
        
        self._warmed_up = False
        self._start_simulation()
        
        return super().reset(**kwargs)
    
    def _sumo_reset(self):
        """
        FIXED: Safer state loading without trying to reset phases
        """
        if self.sumo is None:
            self._start_simulation()
            return
        
        # Try to load saved state
        if os.path.exists(self.state_file):
            try:
                # Just load the state - DON'T try to reset phases
                # The state already has the correct phase!
                self.sumo.simulation.loadState(self.state_file)
                
                # Success - state loaded!
                return
                
            except Exception as e:
                # State load failed, use default reset
                print(f"  (State load failed: {e}, using default reset)")
                pass
        
        # Fallback to default reset
        super()._sumo_reset()
    
    def step(self, action):
        """Step with crash detection"""
        try:
            return super().step(action)
            
        except Exception as e:
            print(f"\n⚠️ Step crashed: {e}")
            self.crash_count += 1
            
            obs = self.observation_space.sample()
            return obs, -10.0, True, True, {'crash': True}




if __name__ == "__main__":
    
    print("="*70)
    print("🚦 PRODUCTION SUMO-RL TRAINING")
    print("="*70)
    print("✅ State saving (fast resets)")
    print("✅ Traffic warmup (stability)")
    print("✅ Crash handling")
    print("✅ Optimized config")
    print("="*70 + "\n")
    
    # Create environment
    env = ProductionSumoEnv(
        net_file='net.net.xml',
        route_file='rou.route.xml',
        
        use_gui=False,
        single_agent=True,
        num_seconds=120,  # SHORT episodes!
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=50,
        
        # CRITICAL crash prevention:
        time_to_teleport=-1,
        max_depart_delay=-1,
        additional_sumo_cmd='--collision.action remove --no-step-log',
        
        # Production features:
        state_file='production_state.xml',
        warmup_steps=120,
        restart_every=50,
    )
    
    # ============================================
    # CLEAR OUTPUT - WHICH LIGHT IS CONTROLLED
    # ============================================
    print(f"✅ Environment created")
    print()
    print(f"{'='*70}")
    print(f"📊 TRAFFIC LIGHT CONTROL CONFIGURATION")
    print(f"{'='*70}")
    print(f"🚦 All traffic signals in network: {env.ts_ids}")
    print(f"   Total lights: {len(env.ts_ids)}")
    print()
    
    if env.single_agent:
        controlled = env.ts_ids[0] if env.ts_ids else "Unknown"
        fixed = env.ts_ids[1:] if len(env.ts_ids) > 1 else []
        
        print(f"🤖 AI CONTROL (Learning):")
        print(f"   ✅ {controlled} - Controlled by DQN Agent")
        print(f"   📊 This light will learn to optimize traffic flow")
        print()
        
        if fixed:
            print(f"🔄 FIXED TIMING (Not Learning):")
            for light in fixed:
                print(f"   ⭕ {light} - Uses default timing program")
            print(f"   📊 These lights will NOT change during training")
        
    else:
        print(f"🤖 MULTI-AGENT MODE:")
        print(f"   All {len(env.ts_ids)} traffic lights controlled by RL")
        for light in env.ts_ids:
            print(f"   ✅ {light}")
    
    print()
    print(f"💡 To see which physical intersection is {env.ts_ids[0]}:")
    print(f"   1. Open: netedit net.net.xml")
    print(f"   2. Look for junction with tl=\"{env.ts_ids[0]}\"")
    print(f"{'='*70}\n")

    
    # Setup
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='dqn_prod',
        verbose=1
    )
    
    # Create model
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
    print("Starting Training")
    print("="*70)
    print("TensorBoard: tensorboard --logdir ./logs/")
    print("="*70 + "\n")
    
    # Train
    try:
        model.learn(
            total_timesteps=100_000,
            callback=checkpoint_callback,
            log_interval=10,
            progress_bar=True,
            tb_log_name="DQN_Production"
        )
        
        model.save("models/dqn_prod_final")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)
        print(f"   Model: models/dqn_prod_final.zip")
        print(f"   Controlled: {env.ts_ids[0] if env.single_agent else 'All lights'}")
        print(f"   Episodes: {env._reset_count}")
        print(f"   Crashes: {env.crash_count}")
        print(f"   Crash rate: {env.crash_count/max(env._reset_count, 1)*100:.1f}%")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        model.save("models/dqn_prod_interrupted")
        print(f"   Episodes: {env._reset_count}")
        print(f"   Crashes: {env.crash_count}")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        model.save("models/dqn_prod_error")""
        
    finally:
        print("\nCleaning up...")
        env.close()
        print("✅ Done")