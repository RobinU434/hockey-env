# hockey-env

This repository contains a hockey-like game environment for RL

## Install

### Basic Installation

``python3 -m pip install git+https://github.com/martius-lab/hockey-env.git``

or add the following line to your Pipfile

``hockey = {editable = true, git = "https://git@github.com/martius-lab/hockey-env.git"}``

### Installation with Multi-Agent Framework Support

To use the environment with PettingZoo or RLlib, install with optional dependencies:

**With PettingZoo:**
```bash
pip install 'git+https://github.com/martius-lab/hockey-env.git#egg=hockey[pettingzoo]'
```

**With RLlib:**
```bash
pip install 'git+https://github.com/martius-lab/hockey-env.git#egg=hockey[rllib]'
```

**With both frameworks:**
```bash
pip install 'git+https://github.com/martius-lab/hockey-env.git#egg=hockey[all]'
```


## HockeyEnv

![Screenshot](assets/hockeyenv1.png)

``hockey.hockey_env.HockeyEnv``

A two-player (one per team) hockey environment.
For our Reinforcment Learning Lecture @ Uni-Tuebingen.
See Hockey-Env.ipynb notebook on how to run the environment.

The environment can be generated directly as an object or via the gym registry:

``env = gym.envs.make("Hockey-v0")``

There is also a version against the basic opponent (with options)

``env = gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)``

### Environment Details

**Observation Space:**
- **Shape:** `Box(-inf, inf, (18,))` - continuous 18-dimensional vector
- **Contents:**
  - Player 1: position (x, y), angle, linear velocity (vx, vy), angular velocity
  - Player 2: position (x, y), angle, linear velocity (vx, vy), angular velocity  
  - Puck: position (x, y), linear velocity (vx, vy)
- In `keep_mode`, two additional dimensions indicate puck possession

**Action Space:**
- **Shape:** `Box(-1, 1, (6,))` - continuous actions for both players (3 per player)
- **Player actions:**
  - `action[0]`: Force in x-direction (translation)
  - `action[1]`: Force in y-direction (translation)
  - `action[2]`: Torque (rotation)
  - In `keep_mode`: `action[3]` controls shooting (4 actions per player, 8 total)
- Actions are applied simultaneously for both players, with clipping to valid ranges
- First 3 dimensions for Player 1, next 3 for Player 2

**Reward Function:**
- **+10** for scoring a goal
- **-10** for conceding a goal
- **Shaped rewards** (can be disabled):
  - Proximity to puck when defending
  - Touch puck reward
  - Puck moving in the right direction

The environment is a **zero-sum game** where Player 2's reward is the negative of Player 1's reward.

## Multi-Agent RL Frameworks

The environment supports synchronous multi-agent RL through two popular frameworks:

### PettingZoo Interface

Install with PettingZoo support:
```bash
pip install 'hockey[pettingzoo]'
```

Use the AEC (Agent-Environment-Cycle) API:
```python
from hockey.interfaces import env
env = env(keep_mode=False)
env.reset()
for agent in env.agent_iter():
    observation = env.observe(agent)
    action = env.action_space(agent).sample()
    env.step(action)
```

### RLlib Interface

Install with RLlib support:
```bash
pip install 'hockey[rllib]'
```

Use with Ray RLlib for distributed training:
```python
from ray.tune.registry import register_env
from hockey.interfaces import make_rllib_env

register_env("hockey", make_rllib_env)
# Configure and train with PPO, DQN, etc.
```

Both interfaces provide a zero-sum two-player game with simultaneous actions.
See [MULTIAGENT.md](MULTIAGENT.md) for detailed documentation and examples.

