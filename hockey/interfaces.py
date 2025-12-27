"""
Multi-agent RL interfaces for the Hockey environment.

This module provides synchronous multi-agent RL interfaces compatible with
both PettingZoo and RLlib frameworks for the Hockey environment.
"""

from typing import Dict, Any, Optional
import numpy as np
from gymnasium import spaces

from hockey.hockey_env import HockeyEnv

# PettingZoo imports
try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import agent_selector, wrappers
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    AECEnv = object  # Dummy base class

# RLlib imports
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as RLlibMultiAgentEnv
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False
    RLlibMultiAgentEnv = object  # Dummy base class


class HockeyPettingZooEnv(AECEnv):
    """
    PettingZoo AEC (Agent Environment Cycle) wrapper for HockeyEnv.
    
    This wrapper provides a synchronous multi-agent interface compatible with
    PettingZoo's API, allowing the hockey environment to be used with PettingZoo-
    compatible algorithms and tools.
    
    Args:
        render_mode: The rendering mode ('human', 'rgb_array', or None)
        keep_mode: Whether to enable keep mode (stick to puck)
        verbose: Whether to print verbose output
        **kwargs: Additional arguments passed to HockeyEnv
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "hockey_pettingzoo_v0",
        "is_parallelizable": False,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        keep_mode: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Create the underlying hockey environment
        self.env = HockeyEnv(
            mode=HockeyEnv.NORMAL,
            keep_mode=keep_mode,
            verbose=verbose
        )
        
        self.render_mode = render_mode
        
        # Define agents
        self.possible_agents = ["player_1", "player_2"]
        self.agents = self.possible_agents[:]
        
        # Agent selector for round-robin turn-taking (both act simultaneously)
        self._agent_selector = agent_selector(self.agents)
        
        # Observation and action spaces
        # Hockey env observation is (18,) and action is (6,) for 2 players
        self.num_actions = 3 if not keep_mode else 4
        
        # Each agent observes the full state
        self.observation_spaces = {
            agent: spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Each agent controls their own player (3 or 4 actions)
        self.action_spaces = {
            agent: spaces.Box(-1, +1, (self.num_actions,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Storage for actions from both agents
        self._actions = {}
        
        # Episode tracking
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        
    def observation_space(self, agent: str) -> spaces.Space:
        """Return observation space for the given agent."""
        return self.observation_spaces[agent]
    
    def action_space(self, agent: str) -> spaces.Space:
        """Return action space for the given agent."""
        return self.action_spaces[agent]
    
    def observe(self, agent: str) -> np.ndarray:
        """
        Return observation for the given agent.
        
        Args:
            agent: The agent identifier
            
        Returns:
            Observation array for the agent
        """
        obs = self._obs
        
        if agent == "player_2":
            # Mirror the observation for player 2
            obs = self.env.obs_agent_two()
            
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed for environment
            options: Additional reset options
        """
        if seed is not None:
            self.env.set_seed(seed)
            
        # Reset the hockey environment
        obs_dict = self.env.reset()
        
        # In hockey_env, reset returns observations for both players
        # obs_dict is typically the observation for player 1
        self._obs = obs_dict
        
        # Reset agent tracking
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Reset episode state
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self._actions = {}
        
        self.num_moves = 0
        
    def step(self, action: np.ndarray):
        """
        Execute one step in the environment.
        
        In the AEC paradigm, agents act sequentially but both actions are
        collected before stepping the environment.
        
        Args:
            action: Action for the current agent
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # If agent is done, just return without acting
            return self._was_dead_step(action)
        
        # Store the action for the current agent
        agent = self.agent_selection
        self._actions[agent] = action
        
        # If we haven't collected actions from all agents yet, move to next agent
        if len(self._actions) < len(self.agents):
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()
            return
        
        # Both agents have acted, now step the environment
        # Combine actions for both players
        combined_action = np.concatenate([
            self._actions["player_1"],
            self._actions["player_2"]
        ])
        
        # Step the underlying hockey environment
        obs, reward, done, trunc, info = self.env.step(combined_action)
        
        self._obs = obs
        self.num_moves += 1
        
        # Update rewards for both agents
        # In hockey, reward is typically given from player 1's perspective
        self.rewards["player_1"] = reward
        self.rewards["player_2"] = -reward  # Zero-sum game
        
        # Update terminations and truncations
        if done:
            # Game ended (goal scored)
            self.terminations = {agent: True for agent in self.agents}
        
        if trunc:
            # Episode truncated (max steps)
            self.truncations = {agent: True for agent in self.agents}
        
        # Update infos
        self.infos = {agent: info for agent in self.agents}
        
        # Clear actions for next round
        self._actions = {}
        
        # Select next agent
        self.agent_selection = self._agent_selector.next()
        
        self._accumulate_rewards()
        
        # Check if all agents are done
        if all(self.terminations.values()) or all(self.truncations.values()):
            self.agents = []
    
    def _was_dead_step(self, action):
        """Handle step when agent is already done."""
        # Agent is already done, so reward is 0
        if self.agent_selection not in self.rewards:
            self.rewards[self.agent_selection] = 0
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
    
    def _accumulate_rewards(self):
        """Accumulate rewards for all agents."""
        for agent in self.agents:
            if agent not in self.rewards:
                self.rewards[agent] = 0
            self._cumulative_rewards[agent] += self.rewards[agent]
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            return self.env.render()
        elif self.render_mode == "rgb_array":
            return self.env.render(mode="rgb_array")
        return None
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def state(self) -> np.ndarray:
        """
        Return the global state of the environment.
        
        Returns:
            Global state observation
        """
        return self._obs


def raw_env(**kwargs) -> HockeyPettingZooEnv:
    """
    Create a raw PettingZoo environment for Hockey.
    
    Args:
        **kwargs: Arguments passed to HockeyPettingZooEnv
        
    Returns:
        Raw HockeyPettingZooEnv instance
    """
    return HockeyPettingZooEnv(**kwargs)


def env(**kwargs) -> AECEnv:
    """
    Create a PettingZoo environment for Hockey with standard wrappers.
    
    This function creates the environment and applies standard PettingZoo wrappers
    for compatibility and error checking.
    
    Args:
        **kwargs: Arguments passed to HockeyPettingZooEnv
        
    Returns:
        Wrapped HockeyPettingZooEnv instance
    """
    if not PETTINGZOO_AVAILABLE:
        raise ImportError(
            "PettingZoo is not installed. Install it with: pip install 'hockey[pettingzoo]'"
        )
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


# ============================================================================
# RLlib MultiAgentEnv Interface
# ============================================================================


class HockeyRLlibEnv(RLlibMultiAgentEnv):
    """
    RLlib MultiAgentEnv wrapper for HockeyEnv.
    
    This wrapper provides a multi-agent interface compatible with RLlib's API,
    allowing the hockey environment to be used with RLlib algorithms.
    
    Both agents act simultaneously in each step (synchronous multi-agent).
    
    Args:
        config: Configuration dictionary that can contain:
            - render_mode: The rendering mode ('human', 'rgb_array', or None)
            - keep_mode: Whether to enable keep mode (stick to puck)
            - verbose: Whether to print verbose output
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not RLLIB_AVAILABLE:
            raise ImportError(
                "RLlib is not installed. Install it with: pip install 'hockey[rllib]'"
            )
        
        super().__init__()
        
        config = config or {}
        
        # Create the underlying hockey environment
        self.env = HockeyEnv(
            mode=HockeyEnv.NORMAL,
            keep_mode=config.get("keep_mode", False),
            verbose=config.get("verbose", False)
        )
        
        self.render_mode = config.get("render_mode", None)
        
        # Define agents (both agents always active in hockey)
        self.agents = self.possible_agents = ["player_1", "player_2"]
        
        # Observation and action spaces
        self.num_actions = 3 if not config.get("keep_mode", False) else 4
        
        # Each agent observes the full state
        self.observation_spaces = {
            agent: spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Each agent controls their own player (3 or 4 actions)
        self.action_spaces = {
            agent: spaces.Box(-1, +1, (self.num_actions,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self._obs = None
        
    def get_observation_space(self, agent_id: str) -> spaces.Space:
        """Return observation space for the given agent."""
        return self.observation_spaces[agent_id]
    
    def get_action_space(self, agent_id: str) -> spaces.Space:
        """Return action space for the given agent."""
        return self.action_spaces[agent_id]
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for environment
            options: Additional reset options
            
        Returns:
            Tuple of (observations_dict, infos_dict) where both are dictionaries
            mapping agent IDs to their respective values
        """
        if seed is not None:
            self.env.set_seed(seed)
        
        # Reset the hockey environment
        obs = self.env.reset()
        self._obs = obs
        
        # Return observations for both agents (simultaneous acting)
        observations = {
            "player_1": obs,
            "player_2": self.env.obs_agent_two()
        }
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(
        self, 
        action_dict: Dict[str, np.ndarray]
    ) -> tuple:
        """
        Execute one step in the environment.
        
        Both agents act simultaneously in each step.
        
        Args:
            action_dict: Dictionary mapping agent IDs to their actions
            
        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
            where each is a dictionary mapping agent IDs to their respective values
        """
        # Combine actions for both players
        combined_action = np.concatenate([
            action_dict["player_1"],
            action_dict["player_2"]
        ])
        
        # Step the underlying hockey environment
        obs, reward, done, trunc, info = self.env.step(combined_action)
        
        self._obs = obs
        
        # Return observations for both agents
        observations = {
            "player_1": obs,
            "player_2": self.env.obs_agent_two()
        }
        
        # Rewards (zero-sum game)
        rewards = {
            "player_1": float(reward),
            "player_2": float(-reward)
        }
        
        # Terminations (goal scored)
        terminateds = {
            "__all__": done
        }
        
        # Truncations (max steps reached)
        truncateds = {
            "__all__": trunc
        }
        
        # Infos
        infos = {
            "player_1": info,
            "player_2": info
        }
        
        return observations, rewards, terminateds, truncateds, infos
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            return self.env.render()
        elif self.render_mode == "rgb_array":
            return self.env.render(mode="rgb_array")
        return None
    
    def close(self):
        """Close the environment."""
        self.env.close()


def make_rllib_env(config: Optional[Dict[str, Any]] = None) -> HockeyRLlibEnv:
    """
    Factory function to create an RLlib-compatible Hockey environment.
    
    This function can be used with RLlib's environment registration:
    
    Example:
        >>> from ray.tune.registry import register_env
        >>> register_env("hockey", make_rllib_env)
        >>> config = PPOConfig().environment("hockey", env_config={
        ...     "keep_mode": False,
        ...     "verbose": False
        ... })
    
    Args:
        config: Configuration dictionary for the environment
        
    Returns:
        HockeyRLlibEnv instance
    """
    return HockeyRLlibEnv(config)
