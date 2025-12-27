# Multi-Agent RL Interfaces for Hockey Environment

This document describes the multi-agent reinforcement learning interfaces available for the Hockey environment, supporting both **PettingZoo** and **RLlib** frameworks.

## Installation

### Base Installation
```bash
pip install hockey
```

### With PettingZoo Support
```bash
pip install 'hockey[pettingzoo]'
```

### With RLlib Support
```bash
pip install 'hockey[rllib]'
```

### With All Multi-Agent Frameworks
```bash
pip install 'hockey[all]'
```

## PettingZoo Interface

The PettingZoo interface provides an AEC (Agent-Environment-Cycle) API for synchronous multi-agent RL.

### Basic Usage

```python
from hockey.interfaces import env as make_pettingzoo_env

# Create environment
env = make_pettingzoo_env(keep_mode=False, verbose=False)

# Reset
env.reset(seed=42)

# Run episode
for agent in env.agent_iter():
    observation = env.observe(agent)
    action = env.action_space(agent).sample()
    env.step(action)
```

### Features
- **Framework**: PettingZoo AEC API
- **Agents**: `player_1`, `player_2`
- **Action Space**: Continuous `Box(-1, 1, (3,))` or `(4,)` with keep_mode
- **Observation Space**: Continuous `Box(-inf, inf, (18,))` per agent
- **Reward Structure**: Zero-sum game

### API Reference

#### `raw_env(**kwargs)`
Creates a raw PettingZoo environment.

**Parameters:**
- `render_mode` (str, optional): Rendering mode ('human', 'rgb_array', or None)
- `keep_mode` (bool): Enable keep mode (stick to puck)
- `verbose` (bool): Print verbose output

#### `env(**kwargs)`
Creates a PettingZoo environment with standard wrappers (recommended).

## RLlib Interface

The RLlib interface provides a MultiAgentEnv API compatible with Ray RLlib for distributed multi-agent training.

### Basic Usage

```python
from hockey.interfaces import make_rllib_env

# Create environment
config = {
    "keep_mode": False,
    "verbose": False,
    "render_mode": None
}
env = make_rllib_env(config)

# Reset
observations, infos = env.reset(seed=42)

# Run episode
done = False
while not done:
    actions = {
        agent: env.get_action_space(agent).sample()
        for agent in env.agents
    }
    observations, rewards, terminateds, truncateds, infos = env.step(actions)
    done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
```

### Training with RLlib

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
from hockey.interfaces import make_rllib_env

# Register environment
register_env("hockey", make_rllib_env)

# Configure algorithm
config = (
    PPOConfig()
    .environment(
        "hockey",
        env_config={"keep_mode": False, "verbose": False}
    )
    .multi_agent(
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
)

# Build and train
algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward={result['env_runners']['episode_reward_mean']}")
```

### Features
- **Framework**: Ray RLlib MultiAgentEnv API
- **Agents**: `player_1`, `player_2` (simultaneous acting)
- **Action Space**: Continuous `Box(-1, 1, (3,))` or `(4,)` with keep_mode per agent
- **Observation Space**: Continuous `Box(-inf, inf, (18,))` per agent
- **Reward Structure**: Zero-sum game
- **Termination**: Uses special `__all__` key for episode termination

### API Reference

#### `HockeyRLlibEnv(config)`
RLlib-compatible multi-agent environment.

**Config Parameters:**
- `render_mode` (str, optional): Rendering mode
- `keep_mode` (bool): Enable keep mode
- `verbose` (bool): Print verbose output

#### `make_rllib_env(config)`
Factory function for environment creation (use with `register_env`).

## Environment Details

### Observation Space
Each agent observes the full state (18-dimensional vector):
- Player 1 position (x, y)
- Player 1 velocity (vx, vy)
- Player 1 angle, angular velocity
- Player 2 position (x, y)
- Player 2 velocity (vx, vy)
- Player 2 angle, angular velocity
- Puck position (x, y)
- Puck velocity (vx, vy)

### Action Space
Each agent controls their player with continuous actions:
- `action[0]`: Linear force in x-direction [-1, 1]
- `action[1]`: Linear force in y-direction [-1, 1]
- `action[2]`: Torque (rotation) [-1, 1]
- `action[3]`: Shoot force (only in keep_mode) [-1, 1]

### Rewards
- Zero-sum game: `reward_player_2 = -reward_player_1`
- +10 for scoring a goal
- -10 for opponent scoring
- Small rewards for puck proximity and movement

### Termination
- Episode ends when a goal is scored
- Episode truncates after maximum timesteps

## Examples

See `examples_multiagent.py` for complete working examples of both interfaces.

Run examples:
```bash
python examples_multiagent.py
```

## Comparison: PettingZoo vs RLlib

| Feature | PettingZoo | RLlib |
|---------|-----------|-------|
| **Best For** | Research, prototyping | Production, distributed training |
| **API Style** | Agent-iterator based | Dictionary-based |
| **Observation Return** | Per-agent via `observe()` | All agents in dict |
| **Action Input** | Per-agent via `step()` | All agents in dict |
| **Training Support** | Manual or third-party | Built-in distributed training |
| **Complexity** | Simple, educational | More complex, scalable |

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{hockey-env,
  author = {Georg Martius},
  title = {Hockey Environment for Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  url = {https://github.com/martius-lab/hockey-env}
}
```

## License

MIT License - See LICENSE file for details.
