"""
Example usage of multi-agent interfaces for Hockey environment.

This script demonstrates how to use both PettingZoo and RLlib interfaces
for the Hockey environment.
"""

import numpy as np


def example_pettingzoo():
    """Example using PettingZoo interface."""
    print("\n=== PettingZoo Interface Example ===\n")
    
    try:
        from hockey.interfaces import env as make_pettingzoo_env
        
        # Create the environment
        env = make_pettingzoo_env(keep_mode=False, verbose=True)
        
        # Reset the environment
        env.reset(seed=42)
        
        # Run a few steps
        for step in range(5):
            # Get current agent
            agent = env.agent_selection
            
            # Sample a random action
            action = env.action_space(agent).sample()
            
            # Step the environment
            env.step(action)
            
            # Get observation and reward for the agent
            if agent in env.agents:
                obs = env.observe(agent)
                reward = env.rewards.get(agent, 0)
                print(f"Step {step}: {agent} acted, reward={reward:.2f}")
            
            # Check if episode is done
            if not env.agents:
                print("Episode finished!")
                break
        
        env.close()
        print("\nPettingZoo example completed successfully!")
        
    except ImportError as e:
        print(f"PettingZoo not available: {e}")
        print("Install with: pip install 'hockey[pettingzoo]'")


def example_rllib():
    """Example using RLlib interface."""
    print("\n=== RLlib Interface Example ===\n")
    
    try:
        from hockey.interfaces import make_rllib_env
        
        # Create the environment
        config = {
            "keep_mode": False,
            "verbose": True,
            "render_mode": None
        }
        env = make_rllib_env(config)
        
        # Reset the environment
        observations, infos = env.reset(seed=42)
        print(f"Initial observations for agents: {list(observations.keys())}")
        
        # Run a few steps
        for step in range(5):
            # Sample random actions for both agents
            actions = {
                agent: env.get_action_space(agent).sample()
                for agent in env.agents
            }
            
            # Step the environment
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            
            print(f"Step {step}:")
            print(f"  Rewards: player_1={rewards['player_1']:.2f}, "
                  f"player_2={rewards['player_2']:.2f}")
            
            # Check if episode is done
            if terminateds.get("__all__", False) or truncateds.get("__all__", False):
                print("Episode finished!")
                break
        
        env.close()
        print("\nRLlib example completed successfully!")
        
    except ImportError as e:
        print(f"RLlib not available: {e}")
        print("Install with: pip install 'hockey[rllib]'")


def example_rllib_with_ray():
    """Example using RLlib with Ray for training."""
    print("\n=== RLlib Training Example ===\n")
    
    try:
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.tune.registry import register_env
        from hockey.interfaces import make_rllib_env
        
        # Register the environment
        register_env("hockey", make_rllib_env)
        
        # Configure the algorithm
        config = (
            PPOConfig()
            .environment(
                "hockey",
                env_config={
                    "keep_mode": False,
                    "verbose": False
                }
            )
            .multi_agent(
                # Map both agents to their respective policies
                policies={"player_1", "player_2"},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id,
            )
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        "player_1": RLModuleSpec(),
                        "player_2": RLModuleSpec(),
                    }
                )
            )
            .resources(num_gpus=0)
            .env_runners(num_env_runners=2)
        )
        
        # Build the algorithm
        algo = config.build()
        
        # Train for a few iterations
        print("Training for 3 iterations...")
        for i in range(3):
            result = algo.train()
            print(f"Iteration {i+1}: "
                  f"episode_reward_mean={result['env_runners']['episode_reward_mean']:.2f}")
        
        algo.stop()
        print("\nRLlib training example completed successfully!")
        
    except ImportError as e:
        print(f"RLlib not fully available: {e}")
        print("Install with: pip install 'hockey[rllib]'")
    except Exception as e:
        print(f"Training error: {e}")


if __name__ == "__main__":
    print("Hockey Multi-Agent RL Interface Examples")
    print("=" * 50)
    
    # Run PettingZoo example
    example_pettingzoo()
    
    # Run RLlib example
    example_rllib()
    
    # Run RLlib training example (optional, commented out by default)
    # Uncomment the line below to run a full training example
    # example_rllib_with_ray()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo install optional dependencies:")
    print("  - PettingZoo: pip install 'hockey[pettingzoo]'")
    print("  - RLlib: pip install 'hockey[rllib]'")
    print("  - Both: pip install 'hockey[all]'")
